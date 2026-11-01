#include "poisson-solver/device_api.h"
int verify_fortran_layout(const poisson_solver_config_v1 *c, size_t bytes) {
    if(bytes != sizeof(*c) || c->version != 1 || c->dimensions != 3 ||
       c->approximation != 4 || c->flags != 7 || c->device_budget_bytes != UINT64_C(0x112233445566))
        return 1;
    for(int i=0;i<3;++i) if(c->sizes[i] != i+2 || c->lengths[i] != .25*(i+1)) return 2;
    for(int i=0;i<6;++i) if(c->boundary_conditions[i] != i) return 3;
    return 0;
}
