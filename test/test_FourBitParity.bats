#!/usr/bin/env bats

BINDIR="@CMAKE_CURRENT_BINARY_DIR@"
SRCDIR="@CMAKE_CURRENT_SOURCE_DIR@"

mkann="${BINDIR}/../bin/wzann-mkann"
train="${BINDIR}/../bin/wzann-train"
repl="${BINDIR}/../bin/wzann-repl"

train_in="${SRCDIR}/mock/FourBitParity-Train.json"
vrfy_in="${SRCDIR}/mock/FourBitParity-Verify.json"


@test "Four-Bit parity checker creation and training" {
    "$mkann" \
        -p wzann::PerceptronNetworkPattern \
        -l 4:ReLU \
        -l 12:Logistic \
        -l 1:Logistic \
        > FourBitParityAnn.in.json
    "$train" \
        -i FourBitParityAnn.in.json \
        -o FourBitParityAnn.out.json \
        -I "$train_in" \
        -V "$vrfy_in" \
        -t wzann::RpropTrainingAlgorithm
    
    parity=$(echo "1 1 0.0 0.0" | "$repl" \
        -i  FourBitParityAnn.out.json)
    [[ "$parity" =~ ^\(0\.9 ]]
}

# vim:ft=sh
