# Chrysalis

**Chrysalis** is a research project exploring how AI agents can **develop evolving personalities through long-term interaction with humans**.

Inspired by the biological chrysalis stage—where transformation occurs—this project investigates how an interactive agent can gradually form **stable behavioral traits and identity through experience**.

The goal is to move beyond static chatbot personalities and build agents that **grow, adapt, and develop behavioral consistency over time**.

---

## Overview

Most AI agents today generate responses based only on short-term context.
As a result, they lack:

* persistent behavioral traits
* long-term adaptation
* consistent identity across interactions

Chrysalis explores a different approach: treating **personality as a learnable and evolving structure** rather than a predefined setting.

Through interaction history, feedback, and memory, the agent gradually develops **stable yet adaptive behavioral tendencies**.

---

## Core Idea

Chrysalis models personality as an evolving internal state shaped by interaction.

```
Human Interaction
        ↓
Interaction Memory
        ↓
Personality Update
        ↓
Personality Representation
        ↓
Agent Behavior
```

Over time, this process allows the agent to form a **consistent behavioral identity**.

---

## Key Components

### Agent Core

LLM-powered interactive agent responsible for dialogue and decision-making.

### Interaction Memory

Stores long-term interaction history and user feedback.

### Trait Core

Latent representation of the agent's behavioral traits.

### Personality Learning

Updates behavioral traits based on interaction experience.

### Human Interface

Communication layer between users and the agent.

---

## Repository Structure

```
chrysalis/

agent/          # core agent logic
personality/    # trait representation
memory/         # interaction memory
learning/       # personality adaptation
interaction/    # human-agent interface
experiments/    # research experiments
docs/           # design notes
```

---

## Architecture

Chrysalis models personality as an evolving internal state influenced by interaction history and feedback.

                    ┌───────────────┐
                    │   Human User  │
                    └───────┬───────┘
                            │
                            ▼
                 ┌────────────────────┐
                 │ Interaction Layer  │
                 │  (chat interface)  │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │      Agent Core    │
                 │   (LLM reasoning)  │
                 └─────────┬──────────┘
                           │
           ┌───────────────┴───────────────┐
           ▼                               ▼
 ┌──────────────────┐            ┌──────────────────┐
 │ Personality Core │            │ Interaction      │
 │  (trait vector)  │            │ Memory           │
 └─────────┬────────┘            └─────────┬────────┘
           │                               │
           └───────────────┬───────────────┘
                           ▼
                ┌────────────────────┐
                │ Personality        │
                │ Learning Module    │
                └────────────────────┘

---

## Research Direction

This project investigates three key questions:

* Can personality **emerge from interaction history**?
* Can agents maintain **stable yet adaptive behavioral traits**?
* Does evolving personality improve **long-term engagement** in human–AI interaction?

---

## Current Status

Early-stage research prototype.

Planned milestones:

* [ ] Personality representation module
* [ ] Interaction memory system
* [ ] Personality learning mechanism
* [ ] Long-term interaction experiments

---

## Vision

Chrysalis explores a simple idea:

> What if AI agents could **grow through interaction**, rather than being statically designed?

By studying evolving behavioral traits, this project aims to better understand **long-term human–AI relationships**.

---

## License

TBD
