from src import config


# ##################################################################
# test frequency bounds
# verify min and max are sensible values
def test_frequency_bounds_are_valid():
    assert config.MIN_FREQUENCY_SECONDS == 300  # 5 minutes
    assert config.MAX_FREQUENCY_SECONDS == 14400  # 4 hours
    assert config.MIN_FREQUENCY_SECONDS < config.MAX_FREQUENCY_SECONDS


# ##################################################################
# test default frequency
# verify default is between min and max
def test_default_frequency_is_valid():
    assert config.MIN_FREQUENCY_SECONDS <= config.DEFAULT_FREQUENCY_SECONDS
    assert config.DEFAULT_FREQUENCY_SECONDS <= config.MAX_FREQUENCY_SECONDS


# ##################################################################
# test server port
# verify port is in valid range
def test_server_port_is_valid():
    assert 1024 < config.SERVER_PORT < 65535
