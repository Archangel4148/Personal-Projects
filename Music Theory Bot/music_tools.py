import dataclasses
import enum
import music21 as m21

# Required musical constants (these account for the tuning and note order)
_SEMITONE_MAP = {  # Positions of notes in an octave
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
    'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
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

def apply_dots(base_duration: float, dot_count: int) -> float:
    total = base_duration
    add = base_duration / 2
    for _ in range(dot_count):
        total += add
        add /= 2
    return total

@dataclasses.dataclass
class Note:
    letter_name: str
    octave: int
    quarter_note_count: float
    tie: str | None = None
    spanners: list[tuple[str, int]] = dataclasses.field(default_factory=list)

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
        # Count dots
        dot_count = 0
        while note_str.endswith("."):
            dot_count += 1
            note_str = note_str[:-1]
        
        # Get the base beat value
        dur_char = note_str[-1]
        if dur_char not in BEAT_COUNT_MAP:
            raise ValueError(f"Invalid duration: {dur_char}")
        base_duration = BEAT_COUNT_MAP[dur_char]
        quarter_note_count = apply_dots(base_duration, dot_count)
        note_str = note_str[:-1]

        tie = None
        if note_str.endswith("~"):
            tie = "start"
            note_str = note_str[:-1]

        octave = DEFAULT_PLAYING_OCTAVE
        if (o := note_str[-1]).isdigit():
            octave = int(o)
            note_str = note_str[:-1]

        return cls(
            letter_name=note_str,
            octave=octave,
            quarter_note_count=quarter_note_count,
            tie=tie
        )
    
    def to_music21(self):
        """
        Convert this Note into a music21.note.Note
        """
        m21_note = m21.note.Note()
        m21_note.duration.quarterLength = self.quarter_note_count

        m21_note.pitch.name = self.letter_name
        m21_note.pitch.octave = self.octave

        # Handle tie (optional)
        if self.tie:
            m21_note.tie = m21.tie.Tie(self.tie)

        # Add metadata for each spanner type
        for sp_type, sp_id in self.spanners:
            if not hasattr(m21_note, "_spanner_tags"):
                m21_note._spanner_tags = []
            m21_note._spanner_tags.append((sp_type, sp_id))

        return m21_note

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
    
    def to_music21(self):
        """
        Convert KeySignature enum to music21.key.Key
        """
        tonic = self.name.split("_")[0]
        return m21.key.Key(tonic, "major")


@dataclasses.dataclass
class Part:
    name: str
    measures: list[dict[str, list[Note]]]
    clef: str = "treble"
    chords: dict[tuple[int, float], str] | None = None

@dataclasses.dataclass
class ScoreData:
    parts: list[Part]
    time_signature: str
    key_signature: KeySignature
    title: str = "Untitled"
    composer: str = "Unknown"

    @classmethod
    def from_string(cls, score_str: str):
        lines = score_str.strip().split("\n")
        
        # Metadata is on the first line
        metadata_line = lines[0]
        metadata = {}
        for item in metadata_line.split("|"):
            if ":" in item:
                key, value = item.split(":", 1)
                metadata[key.strip().lower()] = value.strip()
        title = metadata.get("title", "Untitled")
        composer = metadata.get("composer", "Unknown")

        part_blocks = "\n".join(lines[1:]).split("|")
        parts = []
        for part_block in part_blocks:
            block_lines = [l.strip() for l in part_block.strip().split("\n") if l.strip()]
            if not block_lines: 
                continue

            # Parse the block header
            header = block_lines.pop(0).split()
            part_name, key_str, time_sig = header[0], header[1], header[2]
            clef_type = header[3] if len(header) > 3 else "treble"

            # Build measures
            measures: list[dict[str, list[Note]]] = []
            voice_measure_indices = {}
            current_voice = "v1"
            
            tie_active = {}
            spanner_states: dict[str, int | None] = {
                "slur": None,
                "gliss": None
            }
            spanner_id_counter = 0

            def ensure_measure_exists(idx):
                while len(measures) <= idx:
                    measures.append({})

            for line in block_lines:
                # Handle switching voices
                if line.endswith(":"):
                    current_voice = line[:-1]
                    voice_measure_indices.setdefault(current_voice, 0)
                    tie_active.setdefault(current_voice, False)
                    continue
                    
                voice_measure_indices.setdefault(current_voice, 0)
                tie_active.setdefault(current_voice, False)

                tokens = line.split()
                for token in tokens:
                    if token == "-":
                        voice_measure_indices[current_voice] += 1
                        continue

                    measure_idx = voice_measure_indices[current_voice]
                    ensure_measure_exists(measure_idx)
                    measure = measures[measure_idx]
                    measure.setdefault(current_voice, [])

                    # Tie start
                    if token.startswith("["):
                        note = Note.from_string(token[1:])
                        note.tie = "start"
                        tie_active[current_voice] = True
                    
                    # Tie end
                    elif token.endswith("]"):
                        note = Note.from_string(token[:-1])
                        # End the tie
                        note.tie = "stop"
                        tie_active[current_voice] = False

                    # Slur start
                    elif token.startswith("("):
                        spanner_type = "slur"
                        spanner_id_counter += 1
                        spanner_states[spanner_type] = spanner_id_counter
                        token = token[1:]
                        note = Note.from_string(token)
                        note.spanners.append((spanner_type, spanner_states[spanner_type]))

                    # Slur end
                    elif token.endswith(")"):
                        spanner_type = "slur"
                        token = token[:-1]
                        note = Note.from_string(token)
                        if spanner_states[spanner_type]:
                            note.spanners.append((spanner_type, spanner_states[spanner_type]))
                        spanner_states[spanner_type] = None

                    # Normal note
                    else:
                        note = Note.from_string(token)
                        if tie_active[current_voice]:
                            note.tie = "continue"
                        # add ongoing spanners
                        for sp_type, sp_id in spanner_states.items():
                            if sp_id:
                                note.spanners.append((sp_type, sp_id))

                    # Add the note to the measure
                    measure[current_voice].append(note)

            # Normalize measure counts
            max_measures = max(voice_measure_indices.values(), default=0) + 1
            while len(measures) < max_measures:
                measures.append({})

            parts.append(
                Part(
                    name=part_name,
                    measures=measures,
                    clef=clef_type,
                    chords=None
                )
            )
        return cls(
            parts=parts,
            time_signature=time_sig,
            key_signature=KeySignature.from_string(key_str),
            title=title,
            composer=composer
        )

    def to_music21(self):
        score = m21.stream.Score()
        score.metadata = m21.metadata.Metadata()
        score.metadata.title = self.title
        score.metadata.composer = self.composer

        score.keySignature = self.key_signature.to_music21()

        for part_data in self.parts:
            part_stream = m21.stream.Part()
            part_stream.id = part_data.name

            for i, measure_notes in enumerate(part_data.measures, start=1):
                meas = m21.stream.Measure(number=i)
                # Key + Time signature
                if i == 1:
                    meas.append(self.key_signature.to_music21())
                    meas.append(m21.meter.TimeSignature(self.time_signature))

                    # Clef type
                    if part_data.clef.lower() == "bass":
                        meas.append(m21.clef.BassClef())
                    else:
                        meas.append(m21.clef.TrebleClef())

                bar_duration = m21.meter.TimeSignature(self.time_signature).barDuration.quarterLength

                # Add notes for each voice
                for voice_name, notes in measure_notes.items():
                    voice_stream = m21.stream.Voice()
                    voice_stream.id = voice_name
                    
                    total_duration = 0.0
                    for n in notes:
                        m21_note = n.to_music21()
                        total_duration += m21_note.duration.quarterLength
                        voice_stream.append(m21_note)
                    
                    space_to_fill = bar_duration - total_duration
                    
                    # Fill in empty space with rests
                    if space_to_fill > 0.0001:
                        rest = m21.note.Rest()
                        rest.duration.quarterLength = space_to_fill
                        voice_stream.append(rest)

                    # Check for over-full measures
                    elif space_to_fill < -0.0001:
                        raise ValueError(
                            f"Voice '{voice_name}' in measure {i} exceeds time signature "
                            f"({total_duration} > {bar_duration})"
                        )

                    meas.append(voice_stream)

                # Add chord letters (optional)
                if part_data.chords:
                    for (m_idx, offset), chord_name in part_data.chords.items():
                        if m_idx == i:
                            cs = m21.harmony.ChordSymbol(chord_name)
                            cs.writeAsChord = False
                            meas.insert(offset, cs)
                
                # Add the measure to the part
                part_stream.append(meas)
            
            # Add the part to the score
            score.append(part_stream)

        # Make another pass to insert spanners
        for sp_type in ["slur", "gliss"]:
            spanner_groups: dict[int, list[m21.note.Note]] = {}

            for part in score.parts:
                for n in part.recurse().notes:
                    if hasattr(n, "_spanner_tags"):
                        for stype, sid in n._spanner_tags:
                            if stype == sp_type:
                                spanner_groups.setdefault(sid, []).append(n)

            # Create spanners in music21
            for notes in spanner_groups.values():
                if sp_type == "slur":
                    s = m21.spanner.Slur(notes)
                elif sp_type == "gliss":
                    s = m21.spanner.Glissando(notes)
                score.insert(0, s)
        
        return score