from django.db import models

BLOCKS = [("A", "A"), ("B", "B"), ("C", "C")]


class Participant(models.Model):
    """One participant, identified only by an automatic code.

    variant is 1 (human list first) or 2 (AI 1 list first). No names are stored.
    """
    code = models.CharField(max_length=10, unique=True)
    variant = models.IntegerField(choices=[(1, "Variant 1"), (2, "Variant 2")])
    consent = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return self.code


class Trial(models.Model):
    """One round of one participant in one block.

    trial is 0-based (0-149); the page shows trial + 1. Gestures are coded 0 = rock, 1 = scissors, 2 = paper; outcome is 0 = win, 1 = loss, 2 = draw from the
    participant's view. bot_favorite is empty in block C, where the bot has no favorite gesture. rt_ms is
    measured from the start of the countdown to the last click; timeouts counts missed attempts before the valid choice.
    """
    participant = models.ForeignKey(Participant, on_delete=models.CASCADE)
    block = models.CharField(max_length=1, choices=BLOCKS)
    block_pos = models.IntegerField()
    label = models.CharField(max_length=10, choices=[("Mensch", "Mensch"), ("KI 1", "KI 1"), ("KI 2", "KI 2")])
    trial = models.IntegerField()
    bot_favorite = models.IntegerField(null=True, blank=True)
    bot_move = models.IntegerField()
    choice = models.IntegerField()
    outcome = models.IntegerField()
    rt_ms = models.IntegerField()
    timeouts = models.IntegerField(default=0)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["participant", "block", "trial"], name="unique_trial")]

    def __str__(self) -> str:
        return f"{self.participant.code} {self.block} {self.trial}"


class Note(models.Model):
    """Free-text note a participant writes after one block.

    block_pos is the position of the block in the session (1-3).
    """
    participant = models.ForeignKey(Participant, on_delete=models.CASCADE)
    block_pos = models.IntegerField()
    text = models.TextField(blank=True)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["participant", "block_pos"], name="unique_note")]

    def __str__(self) -> str:
        return f"{self.participant.code} note {self.block_pos}"


class BlockQuestionnaire(models.Model):
    """Answers about the opponent of one block, asked at the end of the session.

    responsiveness is a rating from 1 to 5. The reasons for switching are one yes/no field each;
    reason_other holds the free text for "other" and is empty if not ticked.
    """
    participant = models.ForeignKey(Participant, on_delete=models.CASCADE)
    block = models.CharField(max_length=1, choices=BLOCKS)
    opponent_guess = models.TextField(blank=True)
    responsiveness = models.IntegerField(choices=[(i, str(i)) for i in range(1, 6)])
    noticed = models.TextField(blank=True)
    change_freq = models.CharField(max_length=10, choices=[("never", "nie"), ("1-2", "1–2-mal"), ("3-5", "3–5-mal"), ("6-10", "6–10-mal"), ("always", "ständig")])
    after_loss = models.BooleanField(default=False)
    after_losses = models.BooleanField(default=False)
    opponent_changed = models.BooleanField(default=False)
    not_figured_out = models.BooleanField(default=False)
    opponent_reacted = models.BooleanField(default=False)
    variety = models.BooleanField(default=False)
    intuition = models.BooleanField(default=False)
    dont_remember = models.BooleanField(default=False)
    reason_other = models.TextField(blank=True)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["participant", "block"], name="unique_block_questionnaire")]

    def __str__(self) -> str:
        return f"{self.participant.code} block {self.block}"


class FinalQuestionnaire(models.Model):
    """The two questions asked once at the very end of the session."""
    participant = models.OneToOneField(Participant, on_delete=models.CASCADE)
    ki2_vs_ki1 = models.TextField(blank=True)
    ai_image = models.TextField(blank=True)

    def __str__(self) -> str:
        return f"{self.participant.code} final"
