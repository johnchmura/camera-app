import SwiftUI

struct RuleOfThirdsOverlay: View {
    var body: some View {
        GeometryReader { geo in
            let width = geo.size.width
            let height = geo.size.height
            let oneThirdX = width / 3
            let twoThirdX = 2 * width / 3
            let oneThirdY = height / 3
            let twoThirdY = 2 * height / 3

            ZStack {
                // Draw horizontal dashed lines
                Path { path in
                    path.move(to: CGPoint(x: 0, y: oneThirdY))
                    path.addLine(to: CGPoint(x: width, y: oneThirdY))
                    path.move(to: CGPoint(x: 0, y: twoThirdY))
                    path.addLine(to: CGPoint(x: width, y: twoThirdY))
                }
                .stroke(Color.white.opacity(0.5),
                        style: StrokeStyle(lineWidth: 1, dash: [4, 4]))

                // Draw vertical dashed lines
                Path { path in
                    path.move(to: CGPoint(x: oneThirdX, y: 0))
                    path.addLine(to: CGPoint(x: oneThirdX, y: height))
                    path.move(to: CGPoint(x: twoThirdX, y: 0))
                    path.addLine(to: CGPoint(x: twoThirdX, y: height))
                }
                .stroke(Color.white.opacity(0.5),
                        style: StrokeStyle(lineWidth: 1, dash: [4, 4]))
            }
        }
        .allowsHitTesting(false)
    }
}
