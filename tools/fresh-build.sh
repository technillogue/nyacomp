set -o xtrace
set -o pipefail
set -o errexit
VERSION="0.0.6"
# worked with auditwheel for manywheel torch, but not freshly compiled recent-glibc torch
# nyacomp-$VERSION-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
WHEEL="nyacomp-$VERSION-cp311-cp311-linux_x86_64.whl"
python3.11 setup.py bdist_wheel
rm -rf dist/check || true
mkdir -p dist/check
cd dist/check
unzip -o ../$WHEEL
mkdir nyacomp.libs
cp ../../nvcomp/lib/libnvcomp{,_gdeflate,_bitcomp}.so nyacomp.libs
# this is libnvidia-ml.so.525.125.06 and would be better symlinked, but in general any "libnvidia-ml.so.1" should be enough so leave that as the DT_NEEDED and skip zipfile symlink troubles
cp -av /lib/x86_64-linux-gnu/libnvidia-ml.so.525.125.06 nyacomp.libs/libnvidia-ml.so.1
patchelf --add-rpath '$ORIGIN/nyacomp.libs' _nyacomp.cpython-311-x86_64-linux-gnu.so
patchelf --add-rpath '$ORIGIN/torch/lib' _nyacomp.cpython-311-x86_64-linux-gnu.so
# since we changed rpath, instead of bothering to change the sha and size, just discard that from the record (it is optional)
awk -F, 'BEGIN{OFS=FS} /^_nyacomp.*\.so/ && !c++ { $2 = ""; $3 = ""; gsub(/ /,"")} 1' "nyacomp-$VERSION.dist-info/RECORD" > "nyacomp-$VERSION.dist-info/RECORD"
# add each of the libraries we added to the record, omitting sha and size
find nyacomp.libs -type f | sed 's/$/,,/' >> "nyacomp-$VERSION.dist-info/RECORD"
# if building for release, strip debug symbols from each binary in nyacomp.libs
if [[ "$1" == "release" ]]; then
    find . -name '*.so' | xargs strip --strip-debug # or --strip-unneeded
fi
# recurse, high compression, preserve symlinks, oldest timestamp (should be deterministic builds instead)
zip -r -9 -y -o ../$WHEEL .
