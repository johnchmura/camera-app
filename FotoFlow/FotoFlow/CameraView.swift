import SwiftUI

struct CameraView: View {
    @ObservedObject var cameraModel: CameraModel

    var body: some View {
        ZStack {
            GeometryReader { geometry in
                if cameraModel.isCameraReady {
                    ZStack {
                        // Camera preview
                        CameraPreview(session: cameraModel.getSession())
                            .frame(width: geometry.size.width,
                                   height: geometry.size.width * (4 / 3))
                            .cornerRadius(12)
                            .clipped()
                            .position(x: geometry.size.width / 2,
                                      y: geometry.size.height / 2)
                        
                        // Rule-of-thirds grid overlay
                        RuleOfThirdsOverlay()
                        
                        // T Shape overlay – only visible if zoom is exactly 1x,
                        // no backup is required, a subject is detected,
                        // and the T overlay is enabled.
                        TShapeOverlay(
                            maxFaceWidth: cameraModel.maxFaceWidth,
                            maxBodyHeight: cameraModel.maxBodyHeight,
                            zoomFactor: cameraModel.currentZoomFactor,
                            tColor: cameraModel.tColor,
                            shouldShow: (cameraModel.currentZoomFactor == 1.0 && !cameraModel.backupOverlay && !cameraModel.noSubjectOverlay && cameraModel.tOverlayEnabled)
                        )
                        
                        // Overlay text for "No subject found" or "Please back up"
                        if cameraModel.noSubjectOverlay {
                            Text("No subject found")
                                .foregroundColor(.red)
                                .font(.system(size: 18, weight: .semibold))
                                .padding(8)
                                .background(Color.black.opacity(0.5))
                                .cornerRadius(8)
                                .position(x: geometry.size.width / 2, y: geometry.size.height / 2)
                        } else if cameraModel.backupOverlay {
                            Text("Please back up")
                                .foregroundColor(.yellow)
                                .font(.system(size: 18, weight: .semibold))
                                .padding(8)
                                .background(Color.black.opacity(0.5))
                                .cornerRadius(8)
                                .position(x: geometry.size.width / 2, y: geometry.size.height / 2)
                        }
                    }
                } else {
                    Text("Initializing Camera...")
                        .foregroundColor(.white)
                        .frame(width: geometry.size.width, height: geometry.size.height)
                }
            }
        }
        .background(Color.black.ignoresSafeArea())
        .onAppear {
            cameraModel.startSession()
        }
        .onDisappear {
            cameraModel.stopSession()
        }
    }
}
