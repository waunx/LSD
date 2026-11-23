from rllm_model_gateway.models import GatewayConfig
from rllm_model_gateway.server import create_app


def test_gateway_health_threshold_reaches_session_router():
    app = create_app(
        GatewayConfig(
            health_check_interval=12.5,
            health_failure_threshold=60,
            health_startup_grace_period=240.0,
        )
    )

    router = app.state.router
    assert router._health_interval == 12.5
    assert router._failure_threshold == 60
    assert router._startup_grace_period == 240.0
