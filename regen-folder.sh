#!/bin/bash


### REGEN THE params files given in the release
## go find the sample data in samplesBAK/ and update it in samples/


# blocks
# name="blocks_cylinders-4-flat_20000_None_None_CubeSpaceAE_AMA4Conv_kltune2"

# lightsout digital
# name="lightsout_digital_5_5000_None_None_CubeSpaceAE_AMA4Conv_kltune2"

# lightsout twisted
# name="lightsout_twisted_5_5000_None_None_CubeSpaceAE_AMA4Conv_kltune2"

# puzzle mandrill
# name="puzzle_mandrill_4_4_20000_None_None_CubeSpaceAE_AMA4Conv_kltune2"

# # puzzle mnist
# name="puzzle_mnist_3_3_5000_None_None_CubeSpaceAE_AMA4Conv_kltune2"

# sokoban
name="sokoban_sokoban_image-20000-global-global-2-train_20000_None_None_CubeSpaceAE_AMA4Conv_kltune2"

cd samplesBAK

tar -xf "samples-$name-top5.tar.bz2"

new_name=$(echo $name | sed 's/_None_None//')


echo $new_name

cd samples

mv $name $new_name

cd $new_name

shopt -s extglob

for dir in logs/*/
do

    current_problems_dirr=${dir%*/}

    cd $current_problems_dirr

    echo "pWD"
    pwd

    rm !(net0.h5|aux.json|p_a_z0_net.npz|p_a_z1_net.npz)

    # mkdir -p tmp/checkpoint

    # mv net0.h5 tmp/checkpoint/

    # mv tmp/checkpoint/net0.h5 tmp/checkpoint/weights.h5

    cd ../../

done

shopt -u extglob

cd ..

rm -r ../../samples/$new_name

cp -r $new_name ../../samples/

cd ..

rm -rdf samples


echo "lastPWDD"
pwd

exit 1