using UdonSharp;
using UnityEngine;
using UnityEngine.UI;
using VRC.SDKBase;

namespace WorldLLM {
// Chat panel: prompt field, Send / Stop / Clear history, reply text and status line (Task 05).
// With a controller assigned it runs in shared mode (Task 06): prompts go to the worker, the reply shown is the
// synced response stream, and "Become worker" / "Recover" are available. Without one it drives the local pipeline.
// Buttons call the On*Clicked methods through UdonBehaviour.SendCustomEvent.
[UdonBehaviourSyncMode(BehaviourSyncMode.None)]
public class LlmChatUI : UdonSharpBehaviour {
	public LlmPipeline pipeline;
	public LlmWorkerController controller;
	public LlmResponseStream stream;
	public LlmHistory history;
	public LlmHeartbeat heartbeat;
	public InputField promptField;
	public Text replyText;
	public Text statusText;
	public Text counterText;
	public Text workerText;
	public GameObject becomeButton;   // hidden in the viewer-only (Android) scene
	public GameObject blockButton;    // shown to the worker while answering someone else
	public int maxInputChars = 200;
	// Task 07: preset prompts (sent as they are, so they work without a keyboard) and an optional TextMeshPro
	// input field that can be switched with the legacy InputField to compare IME and paste behaviour.
	public string[] presets = {
		"自己紹介をしてください。",
		"雨の日の過ごし方を三つ教えて。",
		"今日の予定を立てるのを手伝って。",
		"しりとりをしましょう。最初は「りんご」です。",
		"続けてください。",
	};
	public TMPro.TMP_InputField tmpPromptField;   // null when the scene has no Japanese TMP font
	public bool useTmpField;
	private bool locallyCleared;
	private int clearedAtResponse = -1;

	string FieldText() {
		if(useTmpField && tmpPromptField != null)
			return tmpPromptField.text;
		return promptField != null ? promptField.text : "";
	}

	void ClearField() {
		if(useTmpField && tmpPromptField != null)
			tmpPromptField.text = "";
		else if(promptField != null)
			promptField.text = "";
	}

	public void OnSendClicked() {
		if(Submit(FieldText()))
			ClearField();
		Refresh();
	}

	bool Submit(string text) {
		var accepted = controller != null ? controller.SubmitLocal(text) : pipeline.Submit(text);
		if(accepted)
			locallyCleared = false;
		return accepted;
	}

	void SendPreset(int index) {
		if(presets != null && index < presets.Length)
			Submit(presets[index]);
		Refresh();
	}

	public void OnPreset0() { SendPreset(0); }
	public void OnPreset1() { SendPreset(1); }
	public void OnPreset2() { SendPreset(2); }
	public void OnPreset3() { SendPreset(3); }
	public void OnPreset4() { SendPreset(4); }

	public void OnToggleInputClicked() {
		if(tmpPromptField == null || promptField == null)
			return;
		useTmpField = !useTmpField;
		tmpPromptField.gameObject.SetActive(useTmpField);
		promptField.gameObject.SetActive(!useTmpField);
		OnPromptChanged();
		Refresh();
	}

	public void OnStopClicked() {
		if(controller != null)
			controller.RequestStopLocal();
		else
			pipeline.StopGeneration();
		Refresh();
	}

	public void OnClearClicked() {
		if(controller == null) {
			pipeline.ClearHistory();
		} else if(!controller.ClearShared()) {
			// Not the worker: hide the shared conversation on this client only.
			locallyCleared = true;
			clearedAtResponse = stream.responseId;
		}
		Refresh();
	}

	public void OnBecomeWorkerClicked() {
		if(controller != null)
			controller.RequestBecomeWorker();
		Refresh();
	}

	public void OnBlockClicked() {
		if(controller != null)
			controller.BlockCurrentRequester();
		Refresh();
	}

	public void OnRecoverClicked() {
		if(controller != null)
			controller.Recover();
		Refresh();
	}

	public void OnPromptChanged() {
		if(counterText != null)
			counterText.text = FieldText().Length + " / " + maxInputChars + (tmpPromptField != null ? (useTmpField ? "（TMP）" : "（標準）") : "");
	}

	void SetInputEnabled(bool enabled) {
		if(promptField != null)
			promptField.interactable = enabled;
		if(tmpPromptField != null)
			tmpPromptField.interactable = enabled;
	}

	public void OnPipelineUpdate() {
		Refresh();
	}

	public void OnNetworkStateChanged() {
		Refresh();
	}

	void Refresh() {
		if(controller == null)
			RefreshLocal();
		else
			RefreshShared();
	}

	void RefreshLocal() {
		if(replyText != null)
			replyText.text = pipeline.replyText;
		SetInputEnabled(!pipeline.IsBusy());
		if(statusText != null)
			statusText.text = PipelineStatus();
	}

	void RefreshShared() {
		if(locallyCleared && stream.responseId != clearedAtResponse)
			locallyCleared = false;
		if(replyText != null) {
			if(locallyCleared || stream.responseId == 0)
				replyText.text = "";
			else
				replyText.text = stream.requesterName + ": " + stream.prompt + "\n\n" + stream.text
					+ (!stream.done && stream.regenerating ? "\n（ワーカー交代のため再生成中）" : "");
		}
		if(becomeButton != null)
			becomeButton.SetActive(!controller.viewerOnly);
		if(blockButton != null) {
			var local = Networking.LocalPlayer;
			blockButton.SetActive(controller.localWorkerActive && !stream.done && Utilities.IsValid(local) && stream.requesterId != local.playerId);
		}
		if(workerText != null) {
			if(controller.WorkerAbsent())
				workerText.text = heartbeat.workerPhase == LlmHeartbeat.PhaseFailed ? "ワーカー: 障害（復旧待ち）"
					: controller.viewerOnly ? "ワーカーがいません（PC の参加者がワーカーになると使えます）"
					: "ワーカーがいません（PC の方は「ワーカーになる」を押してください）";
			else
				workerText.text = "ワーカー: " + controller.workerName + (controller.localWorkerActive ? "（あなた）" : "");
		}
		// Locked while this player's own prompt is being answered (as in solo mode). A waiting prompt can still be
		// replaced by sending a new one, so the field stays open while waiting.
		SetInputEnabled(controller.LocalQueuePosition() != 0);
		// Restored only into an empty field; otherwise it waits (the user is typing something else).
		if(controller.restoreText.Length > 0 && FieldText().Length == 0) {
			if(useTmpField && tmpPromptField != null)
				tmpPromptField.text = controller.restoreText;
			else if(promptField != null)
				promptField.text = controller.restoreText;
			controller.restoreText = "";
		}
		if(statusText != null) {
			var position = controller.LocalQueuePosition();
			var status = !stream.done ? "生成中 " + stream.requesterName : "待機中";
			if(position > 0)
				status += " / あなたの順番: " + position + " 人待ち";
			else if(position == 0)
				status += " / あなたの質問を処理中";
			status += " / 履歴 " + history.Count() + " 往復";
			if(controller.localMessage.Length > 0)
				status += " / " + controller.localMessage;
			if(controller.localWorkerActive)
				status += " / " + PipelineStatus();
			statusText.text = status;
		}
	}

	string PipelineStatus() {
		var s = pipeline.status;
		if(s == LlmPipeline.StatusLoading)
			return "Loading tokenizer...";
		if(s == LlmPipeline.StatusReady)
			return "Ready";
		if(s == LlmPipeline.StatusGenerating)
			return "Generating... " + pipeline.generatedTokens + " tokens";
		if(s == LlmPipeline.StatusDone)
			return "Done " + pipeline.generatedTokens + " tokens, " + Rate() + " tok/s, turns " + pipeline.turnCount;
		if(s == LlmPipeline.StatusError)
			return "Error";
		return "GPU readback failure";
	}

	string Rate() {
		var dt = pipeline.lastTokenTime - pipeline.firstTokenTime;
		if(pipeline.generatedTokens < 2 || dt <= 0f)
			return "-";
		return ((pipeline.generatedTokens - 1) / dt).ToString("F1");
	}
}
}
