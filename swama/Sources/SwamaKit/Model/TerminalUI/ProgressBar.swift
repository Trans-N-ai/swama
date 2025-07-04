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
        if now.timeIntervalSince(lastPrint) >= 0.1 || (total > 0 && downloaded == total) {
            lastPrint = now
            
            // Handle case where total size is unknown (0)
            guard total > 0 else {
                let downloadedStr = formatBytes(downloaded)
                let speedStr = formatSpeed(speed)
                fputs("\rDownloading... \(downloadedStr) (\(speedStr))", stdout)
                fflush(stdout)
                return
            }
            
            let percent: Double = Double(downloaded) / Double(total)
            // Additional safety check for division result
            guard percent.isFinite else {
                return
            }
            
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
            let speedStr = formatSpeed(speed)
            // Use fputs to stdout for progress bar to behave like print
            fputs("\r[\(bar)] \(percentStr) \(speedStr)", stdout)
            fflush(stdout)
        }
    }
    
    private func formatSpeed(_ speed: Double) -> String {
        if speed > 1024 * 1024 {
            return String(format: "%.2f MB/s", speed / 1024 / 1024)
        }
        else if speed > 1024 {
            return String(format: "%.2f KB/s", speed / 1024)
        }
        else {
            return String(format: "%.0f B/s", speed)
        }
    }
    
    private func formatBytes(_ bytes: Int64) -> String {
        if bytes > 1024 * 1024 {
            return String(format: "%.2f MB", Double(bytes) / 1024 / 1024)
        }
        else if bytes > 1024 {
            return String(format: "%.2f KB", Double(bytes) / 1024)
        }
        else {
            return "\(bytes) B"
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
