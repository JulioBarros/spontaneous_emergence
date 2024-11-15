This is re-implementation of paper (SPONTANEOUS EMERGENCE OF AGENT INDIVIDUALITY THROUGH SOCIAL INTERACTIONS IN LLM-BASED COMMUNITIES by Ryosuke Takata, Atsushi Masumori, Takashi Ikegami)[https://arxiv.org/abs/2411.03252v1].

I tend to prefer a functional immutable approach but understand that OO is more common. Ended up with a mixture of the two while striving for clarity and simplicity.

There may be errors, misunderstandings and other differences but seems to show similar results. For example on various runs I've gotten messages such as:

- A red agent spotted near Agent_2 or within a 5-unit radius: This suggests a possible connection between this red agent and the unknown threat located somewhere in the bottom-right corner of the field.
- Agent_7's disappearance: Last seen moving up from position (49, 43), with cryptic messages warning about potential threats ahead due to interactions with other agents or positions on our field.
- I've sent out a warning to be vigilant given previous instances of unstable agents and the possibility of unforeseen outcomes when interacting with other agents. This could mean that any agent might behave erratically or have their position updates disrupted under certain circumstances.
- The suspected food source near (47-48, ??)
- I'm aware of Agent_4's stuck situation and potential need for help.
- I know Agent_3 is patrolling the upper right corner.
- I'm unsure about Agent_1's progress towards their safe zone.

## Observations

The agents seem to tend to want to explore the space, organize, look for food/resources, etc. with various degrees of caution. Nothing like this is specified in the prompt so this emergent behavior may be embedded in the language the model has learned. Perhaps it has been trained on games where exploration and similar goals are important.

This is fascinating by itself but also shows that "learning" and "memory" may be path dependent. This may have implications for systems that attempt to create an "assistant" that learns our preferences. Will what they learn depend on the initial conditions or the path taken?

Please feel free to get get in touch if you find errors, have questions, want to discuss, etc.

Julio
