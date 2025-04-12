import Foundation
import SwiftData

@Model
final class CapturedPhoto {
    var dateCaptured: Date

    init(dateCaptured: Date = Date()) {
        self.dateCaptured = dateCaptured
    }
}
