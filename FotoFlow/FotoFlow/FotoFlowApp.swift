//
//  FotoFlowApp.swift
//  FotoFlow
//
//  Created by Tashi Bapu on 3/26/25.
//

import SwiftUI
import SwiftData

@main
struct FotoFlowApp: App {
    var sharedModelContainer: ModelContainer = {
        let schema = Schema([
            CapturedPhoto.self,
        ])
        let modelConfiguration = ModelConfiguration(schema: schema, isStoredInMemoryOnly: false)

        do {
            return try ModelContainer(for: schema, configurations: [modelConfiguration])
        } catch {
            fatalError("Could not create ModelContainer: \(error)")
        }
    }()

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .modelContainer(sharedModelContainer)
    }
}
