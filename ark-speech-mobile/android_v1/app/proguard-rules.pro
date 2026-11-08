# sherpa-onnx JNI reads config data-class fields by name.
-keep class com.k2fsa.sherpa.onnx.** { *; }
# ONNX Runtime Java binding.
-keep class ai.onnxruntime.** { *; }
# kotlinx.serialization: keep serialisable models and their generated serializers.
-keepattributes *Annotation*, InnerClasses
-dontnote kotlinx.serialization.AnnotationsKt
-keepclassmembers class kotlinx.serialization.json.** { *** Companion; }
-keepclasseswithmembers class kotlinx.serialization.json.** { kotlinx.serialization.KSerializer serializer(...); }
-keep,includedescriptorclasses class co.immanuel.arkspeech.**$$serializer { *; }
-keep,includedescriptorclasses class co.immanuel.arkspeech.core.**$$serializer { *; }
-keepclassmembers class co.immanuel.arkspeech.** { *** Companion; }
-keepclasseswithmembers class co.immanuel.arkspeech.** { kotlinx.serialization.KSerializer serializer(...); }
