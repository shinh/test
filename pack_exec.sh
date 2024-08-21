#!/bin/bash

e=$1
o=$2

if [ -z $o ]; then
    echo "Usage: $0 exec outdir"
    exit 1
fi

mkdir -p $o

echo "Copying $e (exec)"
cp $e $o
oe=$o/$(basename $e)

ld_linux=
for i in $(ldd $e | grep -o '/[^ ]*'); do
    cp $i $o
    if [[ $i == */ld-linux* ]]; then
        echo "Copying $i (ld-linux)"
        ld_linux=$o/$(basename $i)
    else
        echo "Copying $i (lib)"
    fi
done

cat <<EOF > $o/run.sh
#!/bin/bash

LD_LIBRARY_PATH=$o ${ld_linux} ${oe} "\$@"
EOF

chmod 755 $o/run.sh

echo "Usage: $o/run.sh args..."
