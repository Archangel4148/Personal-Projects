\version "2.24.4"
% automatically converted by musicxml2ly from E:\Programming\Repositories\Personal-Projects\Music Theory Bot\output\sheet_music.musicxml
\pointAndClickOff

\header {
    title =  "Be Thou My Vision"
    subtitle =  "Be Thou My Vision"
    composer =  "Unknown (Irish Melody)"
    encodingsoftware =  "music21 v.9.9.1"
    encodingdate =  "2026-02-14"
    }

#(set-global-staff-size 20.0)
\paper {
    
    }
\layout {
    \context { \Score
        autoBeaming = ##f
        }
    }
PartPFouraFivebFourSevenTwofFiveEightFivefEightbNineZeroTwoEightEightThreefZeroZeroOneaEightSevenfcEightSevenfVoicevOne = 
\relative es' {
    \clef "treble" \time 3/4 \key es \major | % 1
    es4 es4 \stemUp f8 [ \stemUp es8 ] | % 2
    c4 bes4 c4 }

PartPFouraFivebFourSevenTwofFiveEightFivefEightbNineZeroTwoEightEightThreefZeroZeroOneaEightSevenfcEightSevenfVoicevOneChords = 
\chordmode {
    | % 1
    es4 s4 s8 s8 | % 2
    as4 s4 }

PartPFouraFivebFourSevenTwofFiveEightFivefEightbNineZeroTwoEightEightThreefZeroZeroOneaEightSevenfcEightSevenfVoicevTwo = 
\relative bes {
    \clef "treble" \time 3/4 \key es \major | % 1
    bes4 as4 bes4 | % 2
    as4 bes4 c4 }

PartPFouraFivebFourSevenTwofFiveEightFivefEightbNineZeroTwoEightEightThreefZeroZeroOneaEightSevenfcEightSevenfVoiceOne = 
\relative c' {
    \clef "treble" \time 3/4 \key es \major s1. | % 3
    s2 }

PartPeEightdcbcfaFourdEightSevenSevenFiveeEightFiveNinefcdNineNineEightEightbaeSevenNineThreeThreeVoicevThree = 
\relative g {
    \clef "bass" \time 3/4 \key es \major | % 1
    g4 f4 es4 | % 2
    es4 d4 es4 }

PartPeEightdcbcfaFourdEightSevenSevenFiveeEightFiveNinefcdNineNineEightEightbaeSevenNineThreeThreeVoicevFour = 
\relative es, {
    \clef "bass" \time 3/4 \key es \major | % 1
    es4 f4 g4 | % 2
    as4 bes4 as4 }

PartPeEightdcbcfaFourdEightSevenSevenFiveeEightFiveNinefcdNineNineEightEightbaeSevenNineThreeThreeVoiceOne = 
\relative c' {
    \clef "bass" \time 3/4 \key es \major }


% The score definition
\score {
    <<
        
        \context ChordNames = "PartPFouraFivebFourSevenTwofFiveEightFivefEightbNineZeroTwoEightEightThreefZeroZeroOneaEightSevenfcEightSevenfVoicevOneChords" { \PartPFouraFivebFourSevenTwofFiveEightFivefEightbNineZeroTwoEightEightThreefZeroZeroOneaEightSevenfcEightSevenfVoicevOneChords}
        \new Staff
        <<
            
            \context Staff << 
                \mergeDifferentlyDottedOn\mergeDifferentlyHeadedOn
                \context Voice = "PartPFouraFivebFourSevenTwofFiveEightFivefEightbNineZeroTwoEightEightThreefZeroZeroOneaEightSevenfcEightSevenfVoicevOne" {  \voiceOne \PartPFouraFivebFourSevenTwofFiveEightFivefEightbNineZeroTwoEightEightThreefZeroZeroOneaEightSevenfcEightSevenfVoicevOne }
                \context Voice = "PartPFouraFivebFourSevenTwofFiveEightFivefEightbNineZeroTwoEightEightThreefZeroZeroOneaEightSevenfcEightSevenfVoicevTwo" {  \voiceTwo \PartPFouraFivebFourSevenTwofFiveEightFivefEightbNineZeroTwoEightEightThreefZeroZeroOneaEightSevenfcEightSevenfVoicevTwo }
                \context Voice = "PartPFouraFivebFourSevenTwofFiveEightFivefEightbNineZeroTwoEightEightThreefZeroZeroOneaEightSevenfcEightSevenfVoiceOne" {  \voiceThree \PartPFouraFivebFourSevenTwofFiveEightFivefEightbNineZeroTwoEightEightThreefZeroZeroOneaEightSevenfcEightSevenfVoiceOne }
                >>
            >>
        \new Staff
        <<
            
            \context Staff << 
                \mergeDifferentlyDottedOn\mergeDifferentlyHeadedOn
                \context Voice = "PartPeEightdcbcfaFourdEightSevenSevenFiveeEightFiveNinefcdNineNineEightEightbaeSevenNineThreeThreeVoicevThree" {  \voiceOne \PartPeEightdcbcfaFourdEightSevenSevenFiveeEightFiveNinefcdNineNineEightEightbaeSevenNineThreeThreeVoicevThree }
                \context Voice = "PartPeEightdcbcfaFourdEightSevenSevenFiveeEightFiveNinefcdNineNineEightEightbaeSevenNineThreeThreeVoicevFour" {  \voiceTwo \PartPeEightdcbcfaFourdEightSevenSevenFiveeEightFiveNinefcdNineNineEightEightbaeSevenNineThreeThreeVoicevFour }
                \context Voice = "PartPeEightdcbcfaFourdEightSevenSevenFiveeEightFiveNinefcdNineNineEightEightbaeSevenNineThreeThreeVoiceOne" {  \voiceThree \PartPeEightdcbcfaFourdEightSevenSevenFiveeEightFiveNinefcdNineNineEightEightbaeSevenNineThreeThreeVoiceOne }
                >>
            >>
        
        >>
    \layout {}
    % To create MIDI output, uncomment the following line:
    %  \midi {\tempo 4 = 100 }
    }

