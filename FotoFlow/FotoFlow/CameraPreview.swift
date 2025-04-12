import SwiftUI
import AVFoundation

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> UIView {
        let view = PreviewUIView()
        view.setup(session: session)
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {
        guard let previewView = uiView as? PreviewUIView else { return }
        previewView.updateFrame()
    }

    class PreviewUIView: UIView {
        private var previewLayer: AVCaptureVideoPreviewLayer?

        func setup(session: AVCaptureSession) {
            let previewLayer = AVCaptureVideoPreviewLayer(session: session)
            previewLayer.videoGravity = .resizeAspectFill
            layer.addSublayer(previewLayer)
            self.previewLayer = previewLayer
        }

        override func layoutSubviews() {
            super.layoutSubviews()
            previewLayer?.frame = bounds
        }

        func updateFrame() {
            previewLayer?.frame = bounds
        }
    }
}
