package com.example.vadapp

import android.os.Build
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import com.example.vadapp.ui.DevScreen
import com.example.vadapp.ui.MainScreen

class MainActivity : ComponentActivity() {
    private val vm: MainViewModel by viewModels()
    private val devVm: DevViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            val dark = isSystemInDarkTheme()
            val context = LocalContext.current
            val scheme = when {
                Build.VERSION.SDK_INT >= Build.VERSION_CODES.S ->
                    if (dark) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
                dark -> darkColorScheme()
                else -> lightColorScheme()
            }
            var devMode by rememberSaveable { mutableStateOf(false) }
            MaterialTheme(colorScheme = scheme) {
                if (devMode) {
                    DevScreen(
                        devVm,
                        onApply = { model, settings ->
                            vm.applySettings(model, settings)
                            Toast.makeText(this@MainActivity, "Đã áp dụng cho ${model.label}", Toast.LENGTH_SHORT).show()
                        },
                        onBack = { devMode = false },
                    )
                } else {
                    MainScreen(vm, onOpenDev = { vm.player.pause(); devMode = true })
                }
            }
        }
    }

    override fun onStop() {
        super.onStop()
        if (!isChangingConfigurations) vm.player.pause()
    }
}
