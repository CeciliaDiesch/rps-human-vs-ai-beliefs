import json

from django.db import IntegrityError
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from rps_beliefs.bots import outcome
from rps_beliefs.study import BLOCK_INFO, BLOCK_ORDER, load_bot_lists, next_participant

from .models import Participant, Trial

LISTS = load_bot_lists()
N_TRIALS = 150


def start(request: HttpRequest) -> HttpResponse:
    """Create a new participant and remember them in the session."""

    code, variant = next_participant(Participant.objects.count())
    p = Participant.objects.create(code=code, variant=variant)
    request.session["participant_id"] = p.id
    return redirect("play")


def play(request: HttpRequest) -> HttpResponse:
    """Show the game page for the participant's first block.

    Score and next trial are counted from the saved trials, so a reload continues where it stopped.
    """
    p = get_object_or_404(Participant, id=request.session.get("participant_id"))
    block = BLOCK_ORDER[p.variant][0]
    label = BLOCK_INFO[block][0]
    trials = Trial.objects.filter(participant=p, block=block)
    return render(request, "study/play.html", {"label": label, "shown_trial": trials.count() + 1,
                                               "wins": trials.filter(outcome=0).count(), "losses": trials.filter(outcome=1).count(), "n_trials": N_TRIALS})


@require_POST
def play_round(request: HttpRequest) -> JsonResponse:
    """Save one round and return the bot's move and the new score.

    The request body is JSON with shown_trial (1-based, as shown on the page), choice, rt_ms and timeouts.
    The trial is saved 0-based. Answers 400 for an invalid choice and 409 if shown_trial is not the next unsaved one.
    """
    p = get_object_or_404(Participant, id=request.session.get("participant_id"))  # participant from the session (cookie from the browser, data in the database)
    block = BLOCK_ORDER[p.variant][0]                            # first block of the participant's variant (stage 2: one block only)
    label = BLOCK_INFO[block][0]                                 # opponent label shown to the participant
    data = json.loads(request.body)                              # data sent by the browser
    choice = data["choice"]
    shown_trial = data["shown_trial"]                            # round number the browser shows, 1-based
    trials = Trial.objects.filter(participant=p, block=block)    # rounds of this block already saved in the database
    t = trials.count()                                           # current round, 0-based (from the database)
    if choice not in (0, 1, 2):
        return JsonResponse({"error": "invalid choice"}, status=400)
    if shown_trial != t + 1 or t >= N_TRIALS:                    # browser and database disagree, or block already finished
        return JsonResponse({"error": "round out of sync"}, status=409)
    bot = LISTS[BLOCK_INFO[block][1]]                            # bot list of this block ("X" or "Y")
    bot_move = bot["moves"][t]
    bot_favorite = bot["fav"][t]
    try:
        Trial.objects.create(participant=p, block=block, block_pos=1, label=label, trial=t,  # TODO stage 4: block_pos from BLOCK_ORDER
                             bot_favorite=bot_favorite, bot_move=bot_move, choice=choice, outcome=outcome(choice, bot_move),
                             rt_ms=data["rt_ms"], timeouts=data["timeouts"])
    except IntegrityError:                                       # same round sent twice, e.g. double click
        return JsonResponse({"error": "already saved"}, status=409)
    return JsonResponse({"bot_move": bot_move,
                         "wins": trials.filter(outcome=0).count(),     # new query, includes the round just saved
                         "losses": trials.filter(outcome=1).count(),
                         "done": t + 1 == N_TRIALS})
