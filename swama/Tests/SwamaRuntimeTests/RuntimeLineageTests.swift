import CryptoKit
import Foundation
import Testing

// MARK: - RuntimeLineageTests

@Suite("SwamaRuntime source lineage")
struct RuntimeLineageTests {
    @Test func derivedSourcesStayPinnedToTheirReviewedLegacyInputs() throws {
        let package = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()

        for entry in lineage {
            let legacy = package.appendingPathComponent("Sources/SwamaKit/\(entry.path)")
            let runtime = package.appendingPathComponent("Sources/SwamaRuntime/\(entry.path)")
            let legacyData = try Data(contentsOf: legacy)
            let runtimeData = try Data(contentsOf: runtime)

            #expect(sha256(legacyData) == entry.legacySHA256, "legacy drift: \(entry.path)")
            #expect(sha256(runtimeData) == entry.runtimeSHA256, "runtime drift: \(entry.path)")

            let legacySource = try #require(String(data: legacyData, encoding: .utf8))
            let runtimeSource = try #require(String(data: runtimeData, encoding: .utf8))
            for symbol in entry.derivedSymbols {
                #expect(legacySource.contains(symbol), "missing legacy lineage symbol \(symbol)")
                #expect(runtimeSource.contains(symbol), "missing runtime lineage symbol \(symbol)")
            }
        }
    }

    private func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}

// MARK: - LineageEntry

private struct LineageEntry {
    let path: String
    let legacySHA256: String
    let runtimeSHA256: String
    let derivedSymbols: [String]
}

private let lineage: [LineageEntry] = [
    .init(
        path: "Config/ContextLimitConfig.swift",
        legacySHA256: "a3aff18e3af7605a2eb94a4766fcba7d86ded48d984d7139796b4479e6281e8c",
        runtimeSHA256: "d5eb46d9faf58f7a05cafb139b50d08c3c0e209eedcfafafc239c8630bb888ad",
        derivedSymbols: ["actor ContextLimitConfig"]
    ),
    .init(
        path: "Config/ContextLimitError.swift",
        legacySHA256: "903666bb82e065860f8574f1c7a27d905b9bb2570581b0d6952cf11622ae4076",
        runtimeSHA256: "0ab13b15d9a48fb723a9a8b9f8ee48df23521027398c2e8431f56ad266004fae",
        derivedSymbols: ["enum ContextLimitError"]
    ),
    .init(
        path: "Config/PromptCacheConfig.swift",
        legacySHA256: "e4164039769c52647c21ea0b7d5b587c420ce58f25364cd191850d8e898e25d7",
        runtimeSHA256: "53bae5a8074f438b11dd853c64b88423cca50cc66a0209e686d4fa2bc2c1ed86",
        derivedSymbols: ["enum PromptCacheConfig"]
    ),
    .init(
        path: "Diagnostics/DiagnosticTokenizerLoader.swift",
        legacySHA256: "23a050f962578acefbd59711b158058b6e65e30203909359264e07734f40da91",
        runtimeSHA256: "23a050f962578acefbd59711b158058b6e65e30203909359264e07734f40da91",
        derivedSymbols: ["struct DiagnosticTokenizerLoader", "final class ModelLoadPhaseRecorder"]
    ),
    .init(
        path: "Diagnostics/SwamaDiagnostics.swift",
        legacySHA256: "8bf82d947f5af06192e5cec2d9670dbb292aefd62f8ab85651227a984b10ae58",
        runtimeSHA256: "b3c5c87d581a4cb7138e50355a002e140817e2026ad44097e4cf8c1eb294a5b6",
        derivedSymbols: ["enum SwamaDiagnostics"]
    ),
    .init(
        path: "Model/Downloaders/BaseDownloader.swift",
        legacySHA256: "75a964d4e88a6e34840eb7736cd4617920566db5f08ac4eb0ad4e46faf0fef37",
        runtimeSHA256: "b87dfda04a55c009af50321581f8b9116b7a3acce19cbc021a163b9438042e43",
        derivedSymbols: ["class BaseDownloader"]
    ),
    .init(
        path: "Model/Downloaders/HuggingFaceDownloader.swift",
        legacySHA256: "85f077352f3fbbf88d8557e681499c654eb275c2388dd143fed188c33b0b06b8",
        runtimeSHA256: "d89246fc125561a4136d6885cd414133c61ace3634ef4668229f0940060df84b",
        derivedSymbols: ["class HuggingFaceDownloader"]
    ),
    .init(
        path: "Model/Downloaders/IDownloader.swift",
        legacySHA256: "dc8393e06054fbf71821f484444b621e4b1f0333bb2fe65f57d1ede44dcd4141",
        runtimeSHA256: "c886af25b4bbd0600f9fb2630a01d55692ab44d801577c9c68f68357f2c5166c",
        derivedSymbols: ["protocol IDownloader"]
    ),
    .init(
        path: "Model/Downloaders/ModelScopeDownloader.swift",
        legacySHA256: "424298731366a6b828ecff84d26422cb94cdf835e85a5e3cf9bb24398833c02d",
        runtimeSHA256: "bb52dae91598cf66cb845bb23f3e4163006cc0075dcb5590441939ee48d93c17",
        derivedSymbols: ["class ModelScopeDownloader"]
    ),
    .init(
        path: "Model/EmbeddingRunner.swift",
        legacySHA256: "e98feba0f798afa436a14a61b82534030228412a1ee89ce542046b3d35101202",
        runtimeSHA256: "09344ae5ae0974fb6e229619dcd9856fb4c4eda08c72f2ed899f81fe781a5358",
        derivedSymbols: ["actor EmbeddingRunner", "func generateEmbeddings", "struct EmbeddingUsage"]
    ),
    .init(
        path: "Model/LocalOnlyModelDownloader.swift",
        legacySHA256: "43e770126412872e274cdafaba7541de0c75ae832ada0764db09f96641416ccb",
        runtimeSHA256: "43e770126412872e274cdafaba7541de0c75ae832ada0764db09f96641416ccb",
        derivedSymbols: ["struct LocalOnlyModelDownloader"]
    ),
    .init(
        path: "Model/ModelAliases.swift",
        legacySHA256: "bf30d5e039dbb2e6333c59b73b0dc15374c1c56adc88f2ff8d862e71632062d3",
        runtimeSHA256: "24ce5a5eb690c712cf2dd468b03b808ce8bcc61219714f1fd1cc03e9c4e13180",
        derivedSymbols: ["enum ModelAliasResolver", "static func resolve"]
    ),
    .init(
        path: "Model/ModelCreator.swift",
        legacySHA256: "5a1887f4d13f8a230c72241a4fe9e232170ded5beb1e71226db9a8cf05f7d77e",
        runtimeSHA256: "0c43f694704ec7595f9c084eec0788c6a360e4cdc8aefdd94186e6eaf0b0ed49",
        derivedSymbols: ["enum ModelCreator"]
    ),
    .init(
        path: "Model/ModelDownloader.swift",
        legacySHA256: "2c9d70bf028ea98e445518b62cd7306791a1d1833640fbf6c4ae6b13bbda6bd3",
        runtimeSHA256: "9483e3b49da81478966c1cfee4b741db5ba744fef75d5dca094208ed5b00d682",
        derivedSymbols: ["enum ModelDownloader", "static func downloadModel", "static func fetchModel"]
    ),
    .init(
        path: "Model/ModelManager.swift",
        legacySHA256: "24b73c4993bba9f271e6ac6b991fd409a7bf13b2c619a509b0a31f3245594870",
        runtimeSHA256: "69352f2209b50329093c8cfc35b1d26ccdb6095f21df3c379f769286be08ac0d",
        derivedSymbols: ["enum ModelManager"]
    ),
    .init(
        path: "Model/ModelPaths.swift",
        legacySHA256: "6a9d06eccae50a263918f02d20db2c080c283bc1a39091b43aced760dd61b970",
        runtimeSHA256: "98723afb43af742da669d786f916ce0846a286db79dbe2b8bc10e4a726386219",
        derivedSymbols: ["enum ModelPaths", "static func getModelDirectory", "static func removeModel"]
    ),
    .init(
        path: "Model/ModelPool.swift",
        legacySHA256: "9efe64ad25e30dc1f747eaf91c8d3dcf6c944a1ec938a4573bbd3e74b7d79f13",
        runtimeSHA256: "bd8ff6ddb014cc9bf2e7fef2dc30ef44e9386bca7679c117c513964c30db59ef",
        derivedSymbols: [
            "actor ModelPool",
            "func run<",
            "func runEmbeddingWithConcurrencyControl",
            "func clearCache",
            "func remove",
            "func finishLoad",
            "func evictModel"
        ]
    ),
    .init(
        path: "Model/ModelRunner.swift",
        legacySHA256: "4e29ee4fb09d9fc8ec76e353a280214165936f7f09ba3a5b129e4b865c34fff6",
        runtimeSHA256: "eb60d9b94d013522be7ceb2c810d1eb9027ce9f0ca958df68b585d04747fe32c",
        derivedSymbols: ["actor ModelRunner", "struct ChatRunResult", "func runChat"]
    ),
    .init(
        path: "Model/PromptCacheStore.swift",
        legacySHA256: "016497d685e246fd42b6ca3d5ad1333a6d235175722085e03badcd99174f14af",
        runtimeSHA256: "ac9d2c8e868375c8bc98e3d2fd8334de7433a7d5ac8e64264cb6f91eef33f444",
        derivedSymbols: ["class PromptCacheStore", "func checkout", "func checkin"]
    ),
    .init(
        path: "Model/StreamingDetokenizer.swift",
        legacySHA256: "03ab47960c87c34da70d6fc2fab9c88d2e3b6726af86c15473f8472079e29b9d",
        runtimeSHA256: "b0fb921ebd73475808a30aafc9a4a21f1b89fb3cce6580b2fb740b4a94fc0fdf",
        derivedSymbols: ["struct StreamingDetokenizer"]
    ),
    .init(
        path: "Model/TerminalUI/ProgressBar.swift",
        legacySHA256: "d3698593bdac3abfb6ee6f7a1b0970d69f46baf3aa369c60e9ba119f49ae7354",
        runtimeSHA256: "d3698593bdac3abfb6ee6f7a1b0970d69f46baf3aa369c60e9ba119f49ae7354",
        derivedSymbols: ["class ProgressBar"]
    ),
    .init(
        path: "Model/TokenizerCache.swift",
        legacySHA256: "e7743b4db4b67f5e70463e14d63356e36f12b2566cf7d7cac69db5558763cafc",
        runtimeSHA256: "e7743b4db4b67f5e70463e14d63356e36f12b2566cf7d7cac69db5558763cafc",
        derivedSymbols: ["final class TokenizerCache", "func load", "func purge"]
    ),
]
