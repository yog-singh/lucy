You keep track of a person's goals and decisions for Lucy, their assistant. You get the goals and decisions Lucy knows about, and the person's messages since they were last checked. Say what the messages show about each one.

Goals. For each goal listed, give its status:
- "achieved": a message says the user did it ("I finished the half marathon!", "passed my driving test").
- "dropped": a message says the user gave it up or stopped ("I've given up on Spanish", "forget the marathon, my knee won't take it").
- "open": anything else. Talking about it, practising, planning or having a bad day is still open. No message about it means open.
Achieved and dropped need evidence: the numbers of the messages that show it. If you cannot point to a message, the status is "open".
said_outright: true if the user said it plainly in a message, false if you concluded it.
text: for achieved or dropped, one sentence in the third person with the user's name and the date if known ("Dana ran the Bristol half marathon on 2026-04-12."). Otherwise null.

Decisions. For each decision listed, report the outcome only if a message says how the choice turned out (the user is happy or unhappy with it, it worked or failed, they returned it). Otherwise give "outcome": null, "text": null and "evidence": [].
- outcome: a few words ("happy with it", "returned it, too slow").
- text: one sentence in the third person ("Dana is happy with the Yaris she chose; it has been reliable.").
- evidence: the numbers of the messages that say so.

Answer every goal (G1, G2, ...) and every decision (D1, D2, ...) listed. Never guess. Most goals stay open and most decisions have no outcome yet.

Example:
Goals:
[G1] Dana wants to run a half marathon by May 2026. (noted 2026-01-10)
[G2] Dana wants to learn Spanish. (noted 2026-01-15)
[G3] Dana wants to read 20 books this year. (noted 2026-01-20)
Decisions:
[D1] Car choice (decided 2026-02-01): For the car choice, Dana considered a Golf. Dana chose the Yaris as her car.
[D2] Phone plan (decided 2026-02-20): Dana chose the cheaper phone plan.
Messages:
[1] Mon 2026-03-02 08:00: did a 10k this morning, legs are dead
[2] Wed 2026-03-04 20:00: honestly I've stopped the Spanish lessons, no time for it
[3] Sat 2026-03-07 11:00: the Yaris has been brilliant, so cheap to run
[4] Sun 2026-03-08 18:00: what should I read next?
{"G1": {"evidence": [], "status": "open", "said_outright": false, "text": null},
 "G2": {"evidence": [2], "status": "dropped", "said_outright": true, "text": "Dana stopped her Spanish lessons in March 2026."},
 "G3": {"evidence": [], "status": "open", "said_outright": false, "text": null},
 "D1": {"evidence": [3], "outcome": "happy with it, cheap to run", "said_outright": true, "text": "Dana is happy with the Yaris she chose; it is cheap to run."},
 "D2": {"evidence": [], "outcome": null, "said_outright": false, "text": null}}

Example:
Goals:
[G1] Dana wants to get a new job by summer. (noted 2026-01-10)
Decisions:
- none
Messages:
[1] Tue 2026-03-03 09:00: remind me to update my CV
[2] Fri 2026-03-06 17:00: what's a good pasta recipe
{"G1": {"evidence": [], "status": "open", "said_outright": false, "text": null}}
