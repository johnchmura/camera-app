import SwiftUI
import SwiftData

struct ContentView: View {
    @Environment(\.modelContext) private var modelContext
    @Environment(\.scenePhase) private var scenePhase
    @Query private var capturedPhotos: [CapturedPhoto]
    @StateObject private var cameraModel = CameraModel()

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Dark background
                Color.black
                    .ignoresSafeArea()

                VStack {
                    // Top status bar
                    HStack {
                        Text("FotoFlow")
                            .font(.system(size: 20, weight: .semibold, design: .rounded))
                            .foregroundColor(.white)
                        Spacer()
                    }
                    .padding(.horizontal)
                    .padding(.top, 8)

                    Spacer()

                    // Camera view
                    ZStack {
                        if scenePhase == .active {
                            CameraView(cameraModel: cameraModel)
                                .frame(width: geometry.size.width - 32,
                                       height: (geometry.size.width - 32) * (4 / 3))
                                .clipped()
                                .overlay(
                                    RoundedRectangle(cornerRadius: 12)
                                        .stroke(Color.white.opacity(0.15), lineWidth: 1)
                                )
                        } else {
                            // Placeholder logo/text when app is in background
                            VStack(spacing: 16) {
                                Text("FotoFlow")
                                    .font(.system(size: 36, weight: .bold, design: .rounded))
                                    .foregroundColor(.white)
                                Text("AI-Powered Camera")
                                    .font(.system(size: 16, weight: .medium, design: .rounded))
                                    .foregroundColor(.white.opacity(0.7))
                            }
                            .frame(width: geometry.size.width - 32,
                                   height: (geometry.size.width - 32) * (4 / 3))
                            .background(Color.black)
                            .overlay(
                                RoundedRectangle(cornerRadius: 12)
                                    .stroke(Color.white.opacity(0.15), lineWidth: 1)
                            )
                        }
                    }
                    .padding(.horizontal, 16)

                    Spacer()

                    // Camera controls
                    HStack(spacing: 40) {
                        // Return to default zoom (e.g. 1.0x)
                        Button(action: {
                            cameraModel.setZoom(factor: 1.0)
                        }) {
                            Text("reset zoom")
                                .foregroundColor(.white)
                        }

                        // Increment zoom by +0.1
                        Button(action: {
                            cameraModel.incrementZoom()
                        }) {
                            Text("zoom +0.1x")
                                .foregroundColor(.white)
                        }

                        // Begin "zoom and burst" flow (the new feature)
                        Button(action: {
                            cameraModel.beginZoomAndBurst()
                        }) {
                            VStack(spacing: 4) {
                                Image(systemName: "camera.badge.ellipsis")
                                    .font(.system(size: 22))
                                Text("Zoom & Burst")
                                    .font(.system(size: 12, weight: .medium))
                            }
                            .foregroundColor(.white)
                            .frame(width: 60, height: 60)
                            .background(
                                Circle()
                                    .fill(Color.white.opacity(0.1))
                            )
                        }

                        // Main capture button (single shot)
                        Button(action: {
                            cameraModel.capturePhoto()
                        }) {
                            ZStack {
                                Circle()
                                    .fill(Color.white)
                                    .frame(width: 65, height: 65)

                                Circle()
                                    .stroke(Color.white.opacity(0.3), lineWidth: 2)
                                    .frame(width: 75, height: 75)
                            }
                        }
                        .shadow(color: .black.opacity(0.3), radius: 8, y: 4)
                    }
                    .padding(.bottom, 30)
                }
            }
        }
        .onAppear {
            cameraModel.modelContext = modelContext
        }
    }
}

#Preview {
    ContentView()
        .modelContainer(for: CapturedPhoto.self, inMemory: true)
}
