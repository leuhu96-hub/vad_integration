# JNI: native code constructs AnalysisResult and calls ProgressListener.onProgress.
-keep class com.example.vadapp.vad.AnalysisResult { <init>(...); }
-keep interface com.example.vadapp.vad.ProgressListener { *; }
-keepclasseswithmembernames class * { native <methods>; }
