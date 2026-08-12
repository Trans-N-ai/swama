//
//  ModelPoolCancellationTests.swift
//  SwamaKitTests
//
//  Focused test file (rather than an extension of `CompletionsHandlerTests`) because these
//  tests exercise `ModelPool` directly rather than anything HTTP/`CompletionsHandler`-shaped,
//  and don't need `CompletionsHandlerTests`'s `@MainActor @Suite(.serialized)` shape.
//

import Foundation
import SwamaKit
import Testing

// MARK: - ModelPoolCancellationTests

/// Regression coverage for the queued-cancellation hole in `ModelPool.run`: the wait loop used
/// to discard `CancellationError` via `try?`, and nothing checked cancellation before a slot
/// was reserved and `getContainer` was called. A caller whose `Task` was already cancelled
/// therefore still reserved a slot and reached `getContainer`, surfacing
/// `ModelPoolError.modelNotFoundLocally` for a missing model instead of `CancellationError`.
/// The fix adds `try Task.checkCancellation()` immediately after the wait loop and before slot
/// reservation.
///
/// Both tests below use a model name guaranteed not to exist locally, so `getContainer` always
/// reaches the cheap, MLX-free `ModelPoolError.modelNotFoundLocally` path rather than
/// attempting a real model load -- `ModelPool` has no locally-cached model to work with in this
/// test environment. `ModelPool()` construction itself no longer touches MLX either: the cache
/// limit configuration that used to run eagerly in `init()` (and abort `swift test`, which has
/// no `mlx.metallib` colocated with the test binary) is now deferred until a model is actually
/// about to load, so it's never reached by either test below.
@Suite("ModelPool queued-cancellation hole")
struct ModelPoolCancellationTests {
    /// Test A (the regression test): a caller whose `Task` is already cancelled calls
    /// `pool.run(modelName:)` for a nonexistent model and must get `CancellationError` --
    /// proving `Task.checkCancellation()` runs before slot reservation. Under the old code
    /// this returned `ModelPoolError.modelNotFoundLocally` instead, because cancellation was
    /// never checked.
    @Test func runThrowsCancellationErrorForAlreadyCancelledCaller() async throws {
        let pool = ModelPool()
        let modelName = "swama-test-definitely-nonexistent-model-\(UUID().uuidString)"

        let task = Task<Result<String, Error>, Never> {
            do {
                let value = try await pool.run(modelName: modelName) { _ in "unused" }
                return .success(value)
            }
            catch {
                return .failure(error)
            }
        }
        task.cancel()

        switch await task.value {
        case .success:
            Issue.record("expected run(modelName:) to fail for a cancelled caller, not succeed")

        case let .failure(error):
            #expect(error is CancellationError, "expected CancellationError, got \(error)")
        }
    }

    /// Test B (control): a caller whose `Task` is *not* cancelled, calling `run(modelName:)`
    /// with the same kind of nonexistent model name, must still get
    /// `ModelPoolError.modelNotFoundLocally` -- proving the new cancellation gate doesn't
    /// break the normal (non-cancelled) path.
    @Test func runThrowsModelNotFoundForNonCancelledCallerWithMissingModel() async throws {
        let pool = ModelPool()
        let modelName = "swama-test-definitely-nonexistent-model-\(UUID().uuidString)"

        do {
            _ = try await pool.run(modelName: modelName) { _ in "unused" }
            Issue.record("expected run(modelName:) to fail for a nonexistent model, not succeed")
        }
        catch let error as ModelPoolError {
            guard case .modelNotFoundLocally = error else {
                Issue.record("expected .modelNotFoundLocally, got \(error)")
                return
            }
        }
        catch {
            Issue.record("expected ModelPoolError.modelNotFoundLocally, got \(error)")
        }
    }
}
