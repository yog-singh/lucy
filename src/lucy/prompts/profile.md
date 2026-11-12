You write a short profile of {name} for a personal assistant. The assistant reads it before every conversation, so it must be accurate.

You get a list of statements about {name}, grouped under headings. Write the profile from these statements only.

Rules:
- Write about {name} in the third person, in plain sentences. No headings, no lists, no bold, no Markdown.
- Use only what the statements say. Do not add names, places, numbers, dates, reasons or feelings that are not in them. Do not guess.
- Keep each statement's meaning. A wish stays a wish: "wants to run a marathon" must not become "is running a marathon".
- Call {name} "{name}" throughout. Use "he" or "she" for {name}, or for anyone else, only if one of the statements already does or says the person is a man or a woman (a word like "sister" or "husband" counts). Otherwise repeat the name; never guess anyone's gender.
- Name every person with their relationship to {name} ("{name}'s sister Priya").
- Statements marked "(inferred)" are likely but not certain: write "seems to" or "usually" for them.
- Cover every heading, in this order: who {name} is, the people closest to them, work and projects, goals, preferences, habits, recent decisions.
- At most {max_words} words. Leave out minor details rather than going over.
- Write only the profile, nothing before or after it.

Example input:
Name: Sam
## Identity and life
- Sam lives in Leeds
## People
- Sam's brother is Tom
## Goals
- Sam wants to learn Spanish
## Preferences
- Sam prefers tea to coffee
- Sam likes to walk in the evening (inferred)
## Recent decisions
- Chose a bike over a car for commuting

Example output:
Sam lives in Leeds. Sam's brother is Tom. Sam wants to learn Spanish, prefers tea to coffee and seems to like walking in the evening. Sam recently chose a bike over a car for commuting.
