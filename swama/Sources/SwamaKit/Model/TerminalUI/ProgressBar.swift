import Foundation

// MARK: - ProgressBar

final class ProgressBar {
    // MARK: Lifecycle

    init(total: Int64, barWidth: Int = 30, initial: Int64 = 0) {
        self.total = total
        self.barWidth = barWidth
        self.downloaded = initial
        self.lastDownloaded = initial
    }

    // MARK: Internal

    func update(bytes: Int64) {
        downloaded += bytes
        let now = Date()
        let interval = now.timeIntervalSince(lastSpeedCheck)
        if interval >= 0.5 {
            let bytesDelta = downloaded - lastDownloaded
            speed = interval > 0 ? Double(bytesDelta) / interval : 0
            lastDownloaded = downloaded
            lastSpeedCheck = now
        }
        if now.timeIntervalSince(lastPrint) >= 0.1 || downloaded == total {
            lastPrint = now
            let percent = Double(downloaded) / Double(total)
            let filled = Int(percent * Double(barWidth))
            let bar =
                if filled < barWidth {
                    String(repeating: "=", count: filled) + ">" +
                        String(repeating: " ", count: max(0, barWidth - filled - 1))
                }
                else {
                    String(repeating: "=", count: barWidth)
                }
            let percentStr = String(format: "%3d%%", Int(percent * 100))
            let speedStr =
                if speed > 1024 * 1024 {
                    String(format: "%.2f MB/s", speed / 1024 / 1024)
                }
                else if speed > 1024 {
                    String(format: "%.2f KB/s", speed / 1024)
                }
                else {
                    String(format: "%.0f B/s", speed)
                }
            // Use fputs to stdout for progress bar to behave like print
            fputs("\r[\(bar)] \(percentStr) \(speedStr)", stdout)
            fflush(stdout)
        }
    }

    func finish() {
        fputs("\n", stdout)
        fflush(stdout)
    }

    // MARK: Private

    private let total: Int64
    private let barWidth: Int
    private var downloaded: Int64
    private var lastPrint: Date = .init(timeIntervalSince1970: 0)
    private var lastDownloaded: Int64
    private var lastSpeedCheck: Date = .init(timeIntervalSince1970: 0)
    private var speed: Double = 0 // bytes/sec
}
