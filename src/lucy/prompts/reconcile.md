You keep Lucy's memory of the user consistent. A new statement has just been learned. Compare it with the known statements listed and choose one action:

- "reinforce": the new statement says the same thing as a known statement, perhaps in other words. target = that statement's number.
- "supersede": the new statement and a current known statement about the same thing cannot both be true now (the user moved, changed job, switched coffee, gave up a goal). target = the statement it replaces: "lives in York" replaces "lives in Leeds"; "works in York" does not.
- "add": the new statement is different information that can be true alongside everything listed ("lives in Leeds" and "works in York"; "works as a nurse" and "works at St Mary's"; "has a sister Ann" and "has a sister Beth"). When in doubt, add.
- "ignore": the new statement adds nothing worth keeping.

If the new statement says the same as a statement that is no longer true, answer "reinforce" with that number.
Use "target": null for add and ignore.

also_ends: only when you supersede because the user stopped or gave up something, the numbers of the other current statements about the user doing it (a habit of practising it, a goal for it). Otherwise [].

Examples:

Known statements:
[1] Sam drinks dark roast coffee. (preference, stated; current)
New statement: Sam has switched to decaf coffee. (preference, stated)
{"action": "supersede", "target": 1, "also_ends": []}

Known statements:
[1] Sam lives in Leeds. (fact, stated; current)
[2] Sam works as a teacher. (fact, stated; current)
New statement: Sam's home is in Leeds. (fact, stated)
{"action": "reinforce", "target": 1, "also_ends": []}

Known statements:
[1] Sam lives in Leeds. (fact, stated; current)
New statement: Sam works in York. (fact, stated)
{"action": "add", "target": null, "also_ends": []}

Known statements:
[1] Sam lives in Leeds. (fact, stated; no longer true since 2026-02-01, replaced by [2])
[2] Sam lives in York. (fact, stated; current)
New statement: Sam lives in Leeds again. (fact, stated)
{"action": "supersede", "target": 2, "also_ends": []}

Known statements:
[1] Sam wants to learn Spanish. (goal, stated; current)
[2] Sam practises Spanish every evening. (habit, stated; current)
[3] Sam's sister Ann lives in Spain. (fact, stated; current)
New statement: Sam gave up learning Spanish. (fact, stated)
{"action": "supersede", "target": 1, "also_ends": [2]}

Known statements:
[1] Sam's sister Ann lives in Paris. (fact, stated; current)
New statement: Sam's brother Joe lives in Rome. (fact, stated)
{"action": "add", "target": null, "also_ends": []}
