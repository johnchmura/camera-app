import SwiftUI

struct TShapeOverlay: View {
    /// Normalized width (0…1) from the largest detected face bounding box.
    let maxFaceWidth: CGFloat
    /// Normalized height (0…1) from the largest detected body bounding box.
    let maxBodyHeight: CGFloat
    /// Current zoom factor.
    let zoomFactor: CGFloat
    /// Current color of the T shape.
    let tColor: Color
    /// Whether the T shape should be shown.
    let shouldShow: Bool

    var body: some View {
        Group {
            if shouldShow {
                GeometryReader { geo in
                    let horizontalLineWidth = maxFaceWidth * geo.size.width
                    let verticalLineHeight = maxBodyHeight * geo.size.height
                    // Here you can later adjust the vertical position based on area thresholds.
                    let tCenterY = geo.size.height * 0.5
                    let centerX = geo.size.width / 2
                    Path { path in
                        // Horizontal line of the T
                        path.move(to: CGPoint(x: centerX - horizontalLineWidth/2, y: tCenterY))
                        path.addLine(to: CGPoint(x: centerX + horizontalLineWidth/2, y: tCenterY))
                        // Vertical stem of the T
                        path.move(to: CGPoint(x: centerX, y: tCenterY))
                        path.addLine(to: CGPoint(x: centerX, y: tCenterY + verticalLineHeight))
                    }
                    .stroke(tColor, lineWidth: 3)
                }
            } else {
                EmptyView()
            }
        }
        .allowsHitTesting(false)
    }
}
