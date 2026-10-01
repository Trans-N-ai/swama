import Foundation
@preconcurrency import MLX
@testable import SwamaKit
import Testing

// MARK: - TTSWordAlignmentTests

//
// Unit tests for `TTSRunner.alignWords`, the word/duration alignment used by the
// `timestamps: true` speech endpoint (Kokoro today). Pure function, no model load —
// `durations` and `phonemized` are exactly the shapes `KokoroModel.generateWithDurations`
// documents: one duration per token, padded with a leading and trailing zero-duration
// pad token, aligned to `phonemized`'s Unicode scalars.

@Suite("TTS word/duration alignment")
struct TTSWordAlignmentTests {
    @Test
    func alignsTwoWordsSeparatedBySingleSpace() throws {
        // phonemized "ab cd" -> scalars [a, b, ' ', c, d] (5), so durations must carry
        // 5 interior entries + the leading/trailing pad = 7 total.
        let durations = MLXArray([Int32(0), 2, 3, 1, 4, 5, 0])

        // sampleRate 1 makes (sampleCount / sampleRate) / totalFrames come out to exactly
        // 1 second per frame (15 samples over 15 total frames) so expected times are exact.
        let words = try #require(TTSRunner.alignWords(
            inputText: "hi there",
            phonemized: "ab cd",
            durations: durations,
            sampleCount: 15,
            sampleRate: 1
        ))

        #expect(words.count == 2)
        #expect(words[0].text == "hi")
        #expect(words[0].start == 0)
        #expect(words[0].end == 5)
        #expect(words[1].text == "there")
        #expect(words[1].start == 6)
        #expect(words[1].end == 15)

        // Monotonic and never past the clip.
        #expect(words[0].end <= words[1].start)
        #expect(words.last!.end <= 15)
    }

    @Test
    func attachedPunctuationStaysOneWord() throws {
        // "Hello, world!" tokenizes with no whitespace between a word and punctuation
        // that had none in the source text (see EnglishG2P retokenize / MToken.whitespace),
        // so the phonemized string keeps "hello," and "world!" as single space-delimited
        // chunks, exactly like the two whitespace-delimited input words.
        let phonemized = "hi, bye!"
        // scalars: h i , space b y e ! -> 8 scalars, so 10 total durations.
        let durations = MLXArray([Int32(0), 1, 1, 1, 1, 1, 1, 1, 1, 0])

        let words = try #require(TTSRunner.alignWords(
            inputText: "Hi, bye!",
            phonemized: phonemized,
            durations: durations,
            sampleCount: 8,
            sampleRate: 1
        ))

        #expect(words.count == 2)
        #expect(words[0].text == "Hi,")
        #expect(words[1].text == "bye!")
    }

    @Test
    func returnsNilWhenScalarCountDisagreesWithDurationCount() {
        // One too few interior durations for the 5 scalars in "ab cd".
        let durations = MLXArray([Int32(0), 2, 3, 1, 4, 0])

        let words = TTSRunner.alignWords(
            inputText: "hi there",
            phonemized: "ab cd",
            durations: durations,
            sampleCount: 10,
            sampleRate: 1
        )

        #expect(words == nil)
    }

    @Test
    func returnsNilWhenPhonemeWordCountDisagreesWithInputWordCount() {
        // "ab cd ef" phonemizes to three space-delimited chunks; the input text only
        // has two words (e.g. a number that expanded into more phonemized words than
        // the single input token it came from) — alignment must not guess.
        let durations = MLXArray([Int32(0), 1, 1, 1, 1, 1, 1, 1, 1, 0])

        let words = TTSRunner.alignWords(
            inputText: "hi there",
            phonemized: "ab cd ef",
            durations: durations,
            sampleCount: 8,
            sampleRate: 1
        )

        #expect(words == nil)
    }

    @Test
    func returnsNilForEmptyAudio() {
        let durations = MLXArray([Int32(0), 1, 1, 0])

        let words = TTSRunner.alignWords(
            inputText: "hi",
            phonemized: "ab",
            durations: durations,
            sampleCount: 0,
            sampleRate: 24000
        )

        #expect(words == nil)
    }

    @Test
    func returnsNilWhenDurationsAreTooShortToHavePadding() {
        let durations = MLXArray([Int32(0)])

        let words = TTSRunner.alignWords(
            inputText: "hi",
            phonemized: "ab",
            durations: durations,
            sampleCount: 5,
            sampleRate: 1
        )

        #expect(words == nil)
    }
}
