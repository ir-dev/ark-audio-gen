plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.kotlin.serialization)
}

android {
    namespace = "co.immanuel.arkspeech"
    compileSdk = libs.versions.compileSdk.get().toInt()

    defaultConfig {
        applicationId = "co.immanuel.arkspeech"
        minSdk = libs.versions.minSdk.get().toInt()
        targetSdk = libs.versions.targetSdk.get().toInt()
        versionCode = 1
        versionName = "0.1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // sherpa-onnx (static-link variant) and ONNX Runtime both ship arm64-v8a, armeabi-v7a and
        // x86_64. x86 is excluded: the sherpa AAR carries a stray libonnxruntime.so for x86 only,
        // which would collide with ONNX Runtime's own, and no supported device is 32-bit x86.
        // Development builds can ship a single ABI (e.g. -ParkAbi=arm64-v8a), which cuts the debug APK from
        // ~180 MB to ~75 MB; release builds should go out as an App Bundle so Play delivers one ABI anyway.
        val abis = (project.findProperty("arkAbi") as String?)?.split(',')?.map { it.trim() }?.filter { it.isNotEmpty() }
            ?: listOf("arm64-v8a", "armeabi-v7a", "x86_64")
        ndk { abiFilters += abis }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
        debug {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlin {
        compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) }
    }

    buildFeatures { compose = true }

    packaging {
        // Keep .so files uncompressed and page-aligned (16 KB pages on Android 15+).
        jniLibs {
            useLegacyPackaging = false
            // Both the sherpa-onnx static-link AAR and onnxruntime-android carry a lib/x86/libonnxruntime.so;
            // x86 is not a supported ABI here, so drop those libraries instead of picking one.
            excludes += setOf("lib/x86/**")
        }
        resources.excludes += setOf("META-INF/LICENSE*", "META-INF/NOTICE*", "META-INF/*.kotlin_module")
    }

    testOptions {
        unitTests.isIncludeAndroidResources = false
    }

    sourceSets {
        // Sample administrator configuration is bundled as assets (definitions/*.json, routing.json) and
        // seeded on first launch; the test audio is only in the instrumented-test APK.
        getByName("main").assets.srcDirs("../samples")
        getByName("androidTest").assets.srcDirs("../testdata")
    }
}

dependencies {
    implementation(project(":core"))

    // Native speech runtime: sherpa-onnx (Apache-2.0), AAR fetched by scripts/fetch-runtime.sh.
    implementation(files("libs/sherpa-onnx-static-link-onnxruntime-2.0.1.aar"))
    // ONNX Runtime (MIT) for models sherpa-onnx has no loader for (Anuvani-LID).
    implementation(libs.onnxruntime.android)

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.ui.tooling.preview)
    implementation(libs.compose.material3)
    implementation(libs.compose.material.icons.core)
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.kotlinx.serialization.json)
    debugImplementation(libs.compose.ui.tooling)

    testImplementation(libs.junit)
    testImplementation(libs.kotlin.test)
    testImplementation(libs.kotlin.test.junit)
    testImplementation(libs.kotlinx.coroutines.test)

    androidTestImplementation(libs.androidx.test.ext.junit)
    androidTestImplementation(libs.androidx.test.runner)
    androidTestImplementation(libs.androidx.test.rules)
    androidTestImplementation(libs.kotlinx.coroutines.test)
}
