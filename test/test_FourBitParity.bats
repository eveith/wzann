#!/usr/bin/env bats

BINDIR="@CMAKE_CURRENT_BINARY_DIR@"
SRCDIR="@CMAKE_CURRENT_SOURCE_DIR@"

mkann="${BINDIR}/../bin/wzann-mkann"
train="${BINDIR}/../bin/wzann-train"

train_in="${SRCDIR}/mock/FourBitParity-Train.json"
vrfy_in="${SRCDIR}/mock/FourBitParity-Verify.json"


@test "Four-Bit parity checker creation and training" {
    "$mkann" \
        -p wzann::PerceptronNetworkPattern \
        -l 4:ReLU \
        -l 100:Logistic \
        -l 1:Logistic \
        > FourBitParityAnn.in.json
    "$train" \
        -i FourBitParityAnn.in.json \
        -o FourBitParityAnn.out.json \
        -I "$train_in" \
        -V "$vrfy_in" \
        -E 500000 \
        -t wzann::RpropTrainingAlgorithm
}

# vim:ft=sh
