import mlflow

def log_run(result, experiment="starling-base", tracking_uri="sqlite:///mlflow.db"):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    p=result['params']
    metrics={
        "threshold": result["threshold"],
        "auc": result["auc"],
        **result["point"],
        "false_alarm_segments": result["false_alarm_segments"],
        "events_detected": sum(e["detected"] for e in result["events"]),
        "n_events": len(result["events"]),
    }

    with mlflow.start_run(run_name=f"{p['method']}-{p['chan_id']}"):
        mlflow.log_params(p)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v == v})