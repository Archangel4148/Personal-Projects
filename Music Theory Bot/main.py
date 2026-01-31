import dataclasses
import enum

# Required musical constants (these account for the tuning and note order)
_SEMITONE_MAP = {  # Positions of notes in an octave
    'C': 0, 'Cs': 1, 'Db': 1, 'D': 2, 'Ds': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'Fs': 6, 'Gb': 6, 'G': 7, 'Gs': 8,
    'Ab': 8, 'A': 9, 'As': 10, 'Bb': 10, 'B': 11
}
# Frequencies will be tuned to A4 ("equal temperament")
_TUNING_NOTE = "A"
_TUNING_OCTAVE = 4
_TUNING_FREQ = 440.0
_TUNING_SEMITONE_POS = _SEMITONE_MAP[_TUNING_NOTE]

# All notes not specified will be assumed to be in this octave
DEFAULT_PLAYING_OCTAVE = 4

BEAT_COUNT_MAP = {
    "t": 0.125,
    "s": 0.25,
    "e": 0.5,
    "q": 1,
    "h": 2,
    "w": 4,
}

KEY_SHARP_ORDER = "FCGDAEB"
KEY_FLAT_ORDER = KEY_SHARP_ORDER[::-1]
MAJOR_SCALE_NOTES = [0, 2, 4, 5, 7, 9, 11]

@dataclasses.dataclass
class Note:
    letter_name: str
    octave: int
    quarter_note_count: float

    @property
    def frequency(self) -> float:
        # Calculate the frequency from the tuning note (Hz)
        semitone_diff = _SEMITONE_MAP[self.letter_name] - _TUNING_SEMITONE_POS + (
                12 * abs(self.octave - _TUNING_OCTAVE))
        return _TUNING_FREQ * (2 ** (semitone_diff / 12))

    @classmethod
    def from_string(cls, note_str: str):
        """
        note_str should be formatted: "{note letter}{note octave (optional)}{beat count representation}"
        Example: C4h => Note C, octave 4, as a half note (2 beats)
        """
        beat_count = BEAT_COUNT_MAP[note_str[-1]]
        note_str = note_str[:-1]

        octave = DEFAULT_PLAYING_OCTAVE
        if (o := note_str[-1]).isdigit():
            octave = int(o)
            note_str = note_str[:-1]

        return cls(
            letter_name=note_str,
            octave=octave,
            quarter_note_count=beat_count,
        )


class KeySignature(enum.IntEnum):
    # Positive value = sharp count, negative value = flat count
    C_MAJOR = 0
    G_MAJOR = 1
    D_MAJOR = 2
    A_MAJOR = 3
    E_MAJOR = 4
    B_MAJOR = 5
    Fs_MAJOR = 6
    F_MAJOR = -1
    Bb_MAJOR = -2
    Eb_MAJOR = -3
    Ab_MAJOR = -4
    Db_MAJOR = -5
    Gb_MAJOR = -6

    def get_scale(self):
        """Get the notes played in this key's scale (starting from the tonic)"""
        key_note = self.name.split("_")[0]
        all_notes = _SEMITONE_MAP.keys()
        # For flat keys, ignore sharp versions of notes, and vice versa
        if self.value < 0:
            all_notes = [note for note in all_notes if "s" not in note]
        else:
            all_notes = [note for note in all_notes if "b" not in note]
        key_idx = all_notes.index(key_note)
        # Order the notes, starting with the tonic
        ordered_notes = all_notes[key_idx:] + all_notes[:key_idx]
        # Build the scale using the major scale pattern
        scale = []
        for idx in MAJOR_SCALE_NOTES:
            # Handle the exceptions for enharmonics in Fs and Gb
            if self == KeySignature.Fs_MAJOR and idx == 11:
                scale.append("Es")
            elif self == KeySignature.Gb_MAJOR and idx == 4:
                scale.append("Cb")
            else:
                scale.append(ordered_notes[idx])
        return scale

    @classmethod
    def from_string(cls, key_str: str):
        for key_member in KeySignature:
            if key_member.name.split("_")[0] == key_str:
                return key_member
        return None


@dataclasses.dataclass
class Melody:
    measures: list[list[Note]]
    time_signature: str
    key_signature: KeySignature

    @property
    def beats_per_measure(self):
        return int(self.time_signature.split("/")[0])

    @property
    def beats_per_quarter_note(self):
        return float(self.time_signature.split("/")[1]) / 4

    @classmethod
    def from_string(cls, melody_str: str):
        """
        Melody string should start with the key signature, followed by the time signature (Ex: "4/4"),
        and then a list of notes following the format:
        {note string} {note string} - {note string}
        (Notes are separated by spaces, measures are separated by hyphens)
        """
        measure_delimiter = "-"
        tokens = melody_str.split()

        # Read the key signature
        key_str = tokens[0]
        tokens = tokens[1:]

        # Read the time signature
        time_sig_str = tokens[0]
        tokens = tokens[1:]

        # Build the notes from the melody
        measures = [[]] * (melody_str.count(measure_delimiter) + 1)
        current_measure = 0
        for token in tokens:
            if token == measure_delimiter:
                current_measure += 1
                continue
            measures[current_measure].append(Note.from_string(token))

        return cls(
            measures=measures,
            time_signature=time_sig_str,
            key_signature=KeySignature.from_string(key_str)
        )


# 1. Add a melody (If notes are already adjusted for key, is_key_adjusted = True)
is_key_adjusted = True
melody = "C 4/4 Eq Dq Cq Dq - Eq Eq Eh - Dq Dq Dh - Eq Gq Gh - Eq Dq Cq Dq - Eq Eq Eq Eq - Dq Dq Eq Dq - Cw"

# 2. Find the diatonic chords

# 3. Find the cadence notes

# 4. Find the diatonic chords that include each cadence note

# 5. Build a chord progression

# 6. For each chord, find the inversion that's the easiest to play

# melody_obj = Melody.from_string(melody)
# print(melody_obj.beats_per_measure)
# print(melody_obj.beats_per_quarter_note)
# print(melody_obj.key_signature.name, "-", melody_obj.key_signature)
# print(melody_obj.measures)

for key in KeySignature:
    sig = KeySignature.from_string(key.name.split("_")[0])
    print(sig.name, sig.get_scale())
