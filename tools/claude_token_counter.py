"""Compteur de tokens pour l'utilisation de l'API Claude (Anthropic).

Deux usages combinés :

1. Estimation AVANT l'envoi — combien de tokens un prompt va coûter en entrée,
   via l'endpoint officiel `messages.count_tokens` (aucune génération facturée).

2. Suivi de la consommation RÉELLE — on additionne les tokens réellement
   consommés (entrée, sortie, écriture/lecture de cache) à partir du champ
   `usage` renvoyé par chaque réponse de l'API Messages, avec une estimation
   du coût en dollars.

Le comptage de tokens est spécifique au modèle : on passe toujours le même
identifiant de modèle que celui utilisé pour l'inférence. On n'utilise PAS
`tiktoken` (tokenizer d'OpenAI), qui sous-estime fortement les tokens Claude.

Prérequis :
    pip install anthropic
    export ANTHROPIC_API_KEY="sk-ant-..."

Usage rapide :
    python tools/claude_token_counter.py --count "Bonjour Claude"   # estimation
    python tools/claude_token_counter.py --demo                     # appel réel + cumul
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

import anthropic

# Modèle par défaut : Claude Opus 4.8.
DEFAULT_MODEL = "claude-opus-4-8"

# Tarifs publics en dollars par MILLION de tokens (entrée / sortie).
# À garder synchronisé avec https://platform.claude.com/docs/en/pricing
PRICING: dict[str, dict[str, float]] = {
    "claude-fable-5":   {"input": 10.0, "output": 50.0},
    "claude-opus-4-8":  {"input": 5.0,  "output": 25.0},
    "claude-opus-4-7":  {"input": 5.0,  "output": 25.0},
    "claude-opus-4-6":  {"input": 5.0,  "output": 25.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5": {"input": 1.0,  "output": 5.0},
}

# Multiplicateurs de prix pour les tokens de cache (relatifs au prix d'entrée).
# Écriture de cache : 1.25x (TTL 5 min, par défaut) ou 2x (TTL 1 h).
# Lecture de cache  : ~0.1x.
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10

# --- Estimation énergétique (INDICATIVE) -----------------------------------
# ⚠️ Anthropic ne publie PAS de consommation énergétique officielle par token.
# Les valeurs ci-dessous sont des ORDRES DE GRANDEUR issus d'estimations
# publiques sur l'inférence de grands modèles. À ajuster selon vos propres
# mesures. Elles servent à donner une idée, pas un chiffre exact.
#
# La génération (sortie) est séquentielle et bien plus coûteuse par token que
# la lecture du prompt (entrée), traitée en parallèle lors du "prefill".
WH_PER_OUTPUT_TOKEN = 0.0006   # ~2.2 J/token de sortie (estimation indicative)
WH_PER_INPUT_TOKEN = 0.00006   # ~10x moins cher par token d'entrée (prefill)

# Intensité carbone du réseau électrique, en grammes de CO2 par kWh.
# Défaut : moyenne mondiale ~480 g/kWh. À remplacer par la valeur de votre
# région (ex. France ~50, USA ~370) pour affiner.
GRID_CARBON_G_PER_KWH = 480.0


def count_input_tokens(
    client: anthropic.Anthropic,
    messages: list[dict],
    *,
    model: str = DEFAULT_MODEL,
    system: str | None = None,
) -> int:
    """Estime le nombre de tokens d'entrée d'un prompt AVANT de l'envoyer.

    N'effectue aucune génération : utile pour prévoir le coût ou vérifier
    qu'on tient dans la fenêtre de contexte.
    """
    kwargs: dict = {"model": model, "messages": messages}
    if system is not None:
        kwargs["system"] = system
    return client.messages.count_tokens(**kwargs).input_tokens


@dataclass
class UsageTracker:
    """Cumule la consommation réelle de tokens sur plusieurs appels API.

    On lui passe l'objet `response.usage` de chaque réponse de l'API Messages,
    et il maintient les totaux ainsi qu'une estimation du coût en dollars.
    """

    model: str = DEFAULT_MODEL
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    calls: int = 0
    # Liste des coûts par appel, pour un éventuel détail.
    per_call_cost: list[float] = field(default_factory=list)

    def add(self, usage) -> float:
        """Ajoute le `usage` d'une réponse au cumul ; renvoie le coût de cet appel.

        `usage` est l'attribut `response.usage` renvoyé par `messages.create`.
        Les champs de cache valent 0 si le prompt caching n'est pas utilisé.
        """
        self.calls += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        # getattr car ces champs peuvent être absents/None sans prompt caching.
        self.cache_creation_input_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.cache_read_input_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0

        call_cost = self._cost_of(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_creation=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            cache_read=getattr(usage, "cache_read_input_tokens", 0) or 0,
        )
        self.per_call_cost.append(call_cost)
        return call_cost

    @property
    def total_tokens(self) -> int:
        """Total de tous les tokens facturés (entrée + sortie + cache)."""
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )

    @property
    def total_cost(self) -> float:
        """Coût total estimé en dollars sur tous les appels cumulés."""
        return self._cost_of(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cache_creation=self.cache_creation_input_tokens,
            cache_read=self.cache_read_input_tokens,
        )

    @property
    def energy_wh(self) -> float:
        """Énergie estimée consommée (Wh) — INDICATIVE, voir les caveats en tête.

        On compte les tokens d'entrée et de cache comme du "prefill" (peu
        coûteux par token) et les tokens de sortie comme de la génération
        (séquentielle, plus coûteuse).
        """
        input_like = (
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )
        return input_like * WH_PER_INPUT_TOKEN + self.output_tokens * WH_PER_OUTPUT_TOKEN

    @property
    def co2_grams(self) -> float:
        """Émissions de CO2 estimées (grammes) — INDICATIVE."""
        return (self.energy_wh / 1000.0) * GRID_CARBON_G_PER_KWH

    def _cost_of(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cache_creation: int,
        cache_read: int,
    ) -> float:
        rates = PRICING.get(self.model)
        if rates is None:
            # Modèle inconnu : on ne peut pas estimer le coût, on renvoie 0.
            return 0.0
        in_rate = rates["input"] / 1_000_000
        out_rate = rates["output"] / 1_000_000
        return (
            input_tokens * in_rate
            + output_tokens * out_rate
            + cache_creation * in_rate * CACHE_WRITE_MULTIPLIER
            + cache_read * in_rate * CACHE_READ_MULTIPLIER
        )

    def report(self) -> str:
        """Renvoie un récapitulatif lisible de la consommation cumulée."""
        lines = [
            f"Modèle              : {self.model}",
            f"Appels              : {self.calls}",
            f"Tokens entrée       : {self.input_tokens:,}",
            f"Tokens sortie       : {self.output_tokens:,}",
            f"Tokens cache (write): {self.cache_creation_input_tokens:,}",
            f"Tokens cache (read) : {self.cache_read_input_tokens:,}",
            f"Tokens TOTAL        : {self.total_tokens:,}",
            f"Coût estimé         : ${self.total_cost:.4f}",
            f"Énergie estimée*    : {self.energy_wh:.4f} Wh",
            f"CO2 estimé*         : {self.co2_grams:.4f} g",
            "  * estimations indicatives (pas de chiffre officiel Anthropic) ;",
            f"    base : {WH_PER_OUTPUT_TOKEN} Wh/token sortie, {GRID_CARBON_G_PER_KWH:g} gCO2/kWh.",
        ]
        return "\n".join(lines)


def _demo(client: anthropic.Anthropic, model: str) -> None:
    """Fait un vrai appel API et montre l'estimation + le cumul réel."""
    tracker = UsageTracker(model=model)
    messages = [{"role": "user", "content": "Explique le théorème de Pythagore en une phrase."}]

    # 1) Estimation AVANT l'envoi.
    estimated = count_input_tokens(client, messages, model=model)
    print(f"Estimation des tokens d'entrée (avant envoi) : {estimated:,}\n")

    # 2) Appel réel + suivi de la consommation.
    response = client.messages.create(model=model, max_tokens=1024, messages=messages)
    cost = tracker.add(response.usage)

    text = next((b.text for b in response.content if b.type == "text"), "")
    print(f"Réponse : {text}\n")
    print(f"Coût de cet appel : ${cost:.6f}\n")
    print(tracker.report())


def main() -> None:
    parser = argparse.ArgumentParser(description="Compteur de tokens pour l'API Claude.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Identifiant du modèle.")
    parser.add_argument("--count", metavar="TEXTE", help="Estime les tokens d'un texte (sans génération).")
    parser.add_argument("--demo", action="store_true", help="Appel réel + cumul de la consommation.")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        parser.error("Définis la variable d'environnement ANTHROPIC_API_KEY.")

    client = anthropic.Anthropic()

    if args.count is not None:
        n = count_input_tokens(client, [{"role": "user", "content": args.count}], model=args.model)
        print(f"{n:,} tokens d'entrée (modèle {args.model})")
    elif args.demo:
        _demo(client, args.model)
    else:
        parser.error("Précise --count \"...\" ou --demo.")


if __name__ == "__main__":
    main()
