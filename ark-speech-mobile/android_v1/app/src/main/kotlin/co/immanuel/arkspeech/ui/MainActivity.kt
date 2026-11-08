package co.immanuel.arkspeech.ui

import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Build
import androidx.compose.material.icons.filled.Call
import androidx.compose.material.icons.filled.List
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.lifecycle.compose.collectAsStateWithLifecycle

class MainActivity : ComponentActivity() {
    private val vm: MainViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { ArkTheme { ArkApp(vm) } }
    }

    override fun onResume() {
        super.onResume()
        // Models may have been pushed or downloaded while we were away: refresh what the screens show.
        vm.bumpVersion()
    }

    override fun onStop() {
        super.onStop()
        // Push-to-talk never records in the background.
        if (vm.recorder.isRecording) vm.cancelRecording()
        window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }
}

@Composable
fun ArkApp(vm: MainViewModel) {
    val state by vm.state.collectAsStateWithLifecycle()
    val snackbar = remember { SnackbarHostState() }
    LaunchedEffect(state.message) {
        state.message?.let { snackbar.showSnackbar(it); vm.dismissMessage() }
    }
    Scaffold(
        snackbarHost = { SnackbarHost(snackbar) },
        bottomBar = {
            NavigationBar {
                NavigationBarItem(selected = state.screen == Screen.RECORD, onClick = { vm.navigate(Screen.RECORD) }, icon = { Icon(Icons.Default.Call, null) }, label = { Text("Record") })
                NavigationBarItem(selected = state.screen == Screen.MODELS, onClick = { vm.navigate(Screen.MODELS) }, icon = { Icon(Icons.Default.Build, null) }, label = { Text("Models") })
                NavigationBarItem(selected = state.screen == Screen.ADMIN, onClick = { vm.navigate(Screen.ADMIN) }, icon = { Icon(Icons.Default.List, null) }, label = { Text("Admin") })
                NavigationBarItem(selected = state.screen == Screen.SETTINGS, onClick = { vm.navigate(Screen.SETTINGS) }, icon = { Icon(Icons.Default.Settings, null) }, label = { Text("Settings") })
            }
        },
    ) { padding ->
        val m = Modifier.padding(padding)
        when (state.screen) {
            Screen.RECORD -> RecordFlow(vm, state, m)
            Screen.MODELS -> ModelsScreen(vm, state, m)
            Screen.ADMIN -> AdminScreen(vm, state, m)
            Screen.SETTINGS -> SettingsScreen(vm, state, m)
        }
    }
}
