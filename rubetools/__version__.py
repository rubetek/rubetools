# -*- coding: utf-8 -*-

VERSION = (1, 0, 2)
PRERELEASE = None  # alpha, beta or rc
REVISION = None


def generate_version(version, prerelease=None, revision=None):
    version_parts = [".".join(map(str, version))]
    if prerelease is not None:
        version_parts.append("-{}".format(prerelease))
    if revision is not None:
        version_parts.append(".{}".format(revision))
    return "".join(version_parts)


__title__ = "rubetools"
__version__ = generate_version(VERSION, prerelease=PRERELEASE, revision=REVISION)
__author__ = "RubetekAI"
__description__ = "Rubetek dataset tools."
__url__ = "https://github.com/rubetek/rubetools"
__license__ = "Apache License 2.0"
__keywords__ = "rubetools rubetek datatools ai"
