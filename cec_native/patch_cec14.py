"""Turn the official CEC2014 C source into a Linux shared library.

The same five mechanical edits that `patched_cec17_test_func.cpp` carries, applied
to `cec14_test_func.cpp`.  Kept as a script rather than a hand-edited copy so the
diff against the competition source is auditable:

1. Windows-only headers removed (`WINDOWS.H`, `malloc.h`); `stdlib.h`/`string.h`
   added for the allocation and string functions they used to pull in.
2. The globals the official `main.cpp` defines are defined here instead, so the
   file links as a standalone library.
3. The hard-coded `input_data/` prefix becomes a settable directory, so the data
   can live anywhere.
4. **`%Lf` -> `%lf`.**  This is not cosmetic.  The official source reads its
   rotation and shift matrices with `fscanf(fpt,"%Lf",&M[i])` where `M` is
   `double*`.  Under glibc `%Lf` means `long double`, so every matrix entry is
   parsed into the wrong width and the suite returns inf/nan.  The same bug is
   in the CEC2017 source and is patched there identically.
5. `cec14_test_func` gets C linkage so ctypes can find it.

Run:  python3 patch_cec14.py <official cec14_test_func.cpp>
"""
import re
import sys

src = open(sys.argv[1], encoding="latin-1").read()
edits = []


def sub(pattern, repl, what, count=0):
    global src
    new, n = re.subn(pattern, repl, src, count=count)
    assert n > 0, f"no match for {what}"
    src = new
    edits.append(f"{what}: {n}")


sub(r"#include <WINDOWS\.H>\s*\n", "", "remove WINDOWS.H")
sub(r"#include <malloc\.h>", "#include <stdlib.h>\n#include <string.h>", "malloc.h -> stdlib/string")
sub(r"extern double \*OShift,\*M,\*y,\*z,\*x_bound;\s*\nextern int ini_flag,n_flag,func_flag,\*SS;",
    """/* ---- harness additions (SEHHO-COBL reproducibility package) ---- */
extern "C" {
char g_data_dir[4096] = "input_data";
double *OShift=0,*M=0,*y=0,*z=0,*x_bound=0;
int ini_flag=0,n_flag=0,func_flag=0,*SS=0;
void cec_set_data_dir(const char* d){ strncpy(g_data_dir, d, sizeof(g_data_dir)-1); g_data_dir[sizeof(g_data_dir)-1]=0; ini_flag=0; }
void cec_reset(void){ ini_flag=0; }
}""", "globals defined locally")
sub(r'sprintf\(FileName, "input_data/', 'sprintf(FileName, "%s/', "data dir placeholder")
# the placeholder needs g_data_dir as the first vararg
sub(r'sprintf\(FileName, "%s/M_%d_D%d\.txt", func_num,nx\);',
    'sprintf(FileName, "%s/M_%d_D%d.txt", g_data_dir, func_num,nx);', "M path arg")
sub(r'sprintf\(FileName, "%s/shift_data_%d\.txt", func_num\);',
    'sprintf(FileName, "%s/shift_data_%d.txt", g_data_dir, func_num);', "shift path arg")
sub(r'sprintf\(FileName, "%s/shuffle_data_%d_D%d\.txt", func_num, nx\);',
    'sprintf(FileName, "%s/shuffle_data_%d_D%d.txt", g_data_dir, func_num, nx);', "shuffle path arg")
sub(r'"%Lf"', '"%lf"', "%Lf -> %lf (the glibc long-double bug)")
sub(r"\nvoid cec14_test_func\(", '\nextern "C" void cec14_test_func(', "C linkage", count=1)

open("patched_cec14_test_func.cpp", "w").write(src)
print("wrote patched_cec14_test_func.cpp")
for e in edits:
    print("  ", e)
