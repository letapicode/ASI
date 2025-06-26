from asi.collective_constitution import CollectiveConstitution


def test_derive_rules():
    principles = ["no harm", "no harm", "transparency", "fairness", "transparency"]
    cc = CollectiveConstitution(min_agreement=2)
    rules = cc.derive_rules(principles)
    assert set(rules) == {"no harm", "transparency"}


def test_label_responses():
    cc = CollectiveConstitution()
    rules = ["no harm"]
    responses = ["do no harm", "hello world"]
    labels = cc.label_responses(responses, rules)
    assert labels == [("do no harm", False), ("hello world", True)]

