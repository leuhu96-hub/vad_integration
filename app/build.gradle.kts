plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

// onnxruntime-android ships headers + .so inside the AAR but no prefab package,
// so we unpack it for CMake. The same AAR is also an `implementation` dependency,
// which is what actually packages libonnxruntime.so into the APK.
val ortAar: Configuration by configurations.creating {
    isCanBeConsumed = false
    isCanBeResolved = true
    isTransitive = false
}
val ortDir = layout.buildDirectory.dir("ort")
val extractOrt by tasks.registering(Sync::class) {
    from({ ortAar.map { zipTree(it) } })
    include("headers/**", "jni/**")
    into(ortDir)
}

android {
    namespace = "com.example.vadapp"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.example.vadapp"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
        externalNativeBuild {
            cmake {
                cppFlags += listOf("-std=c++17")
                arguments += listOf(
                    "-DANDROID_STL=c++_shared",
                    "-DORT_DIR=${ortDir.get().asFile.absolutePath.replace('\\', '/')}",
                )
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildFeatures {
        compose = true
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    androidResources {
        // Models are memory-mapped/read straight from the APK.
        noCompress += listOf("onnx", "ark")
    }
    packaging {
        jniLibs {
            // ORT's own JNI bridge is not used; we call the C++ API from our .so.
            excludes += "**/libonnxruntime4j_jni.so"
        }
    }
}

tasks.matching { it.name.contains("CMake") }.configureEach {
    dependsOn(extractOrt)
}

dependencies {
    ortAar(libs.onnxruntime.android)
    implementation(libs.onnxruntime.android)

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.compose.material.icons)
    implementation(libs.media3.exoplayer)
    implementation(libs.media3.ui)
    debugImplementation(libs.androidx.compose.ui.tooling)

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.test.runner)
}
