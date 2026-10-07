from bgp_fate.mrt import as_path, communities, prefix_text, update_nlri


def test_normalizes_as_path_communities_and_ipv4_nlri():
    attrs = [
        {"type": {2: "AS_PATH"}, "value": [{"value": [64500, 64501]}]},
        {"type": {8: "COMMUNITY"}, "value": ["64500:1"]},
    ]
    message = {
        "nlri": [{"prefix": "192.0.2.0", "length": 24}],
        "withdrawn_routes": [{"prefix": "198.51.100.0", "length": 24}],
        "path_attributes": attrs,
    }
    announcements, withdrawals = update_nlri(message)
    assert as_path(attrs) == (64500, 64501)
    assert communities(attrs) == ("64500:1",)
    assert prefix_text(announcements[0]) == "192.0.2.0/24"
    assert prefix_text(withdrawals[0]) == "198.51.100.0/24"


def test_normalizes_multiprotocol_nlri():
    message = {
        "nlri": [],
        "withdrawn_routes": [],
        "path_attributes": [
            {
                "type": {14: "MP_REACH_NLRI"},
                "value": {"nlri": [{"prefix": "2001:db8::", "length": 32}]},
            },
            {
                "type": {15: "MP_UNREACH_NLRI"},
                "value": {"withdrawn_routes": [{"prefix": "2001:db8:1::", "length": 48}]},
            },
        ],
    }
    announcements, withdrawals = update_nlri(message)
    assert prefix_text(announcements[0]) == "2001:db8::/32"
    assert prefix_text(withdrawals[0]) == "2001:db8:1::/48"
