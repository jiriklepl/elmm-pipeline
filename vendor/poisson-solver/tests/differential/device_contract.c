#include "poisson-solver/device_api.h"
#include <stdio.h>
int main(void) {
    poisson_solver_handle_v1 *h=(void*)1;
    poisson_solver_config_v1 c={0};
    int started=99;
    if (poisson_solver_create_v1(&c,&h)!=POISSON_SOLVER_ARGUMENT || h) return 1;
    c.version=1; c.dimensions=4; c.approximation=2;
    if (poisson_solver_create_v1(&c,&h)!=POISSON_SOLVER_UNSUPPORTED || h) return 2;
    c.dimensions=3; c.device_budget_bytes=1;
    for(int i=0;i<3;++i) { c.sizes[i]=2; c.lengths[i]=1; }
    if (poisson_solver_create_v1(&c,&h)!=POISSON_SOLVER_RESOURCE || h) return 7;
    if (poisson_solver_execute_device_v1(0,0,0,0,0,0,0,0,&started)!=POISSON_SOLVER_ARGUMENT || started) return 3;
    if (poisson_solver_workspace_query_v1(0,0)!=POISSON_SOLVER_ARGUMENT) return 4;
    if (poisson_solver_destroy_v1(&h)!=POISSON_SOLVER_OK || h) return 5;
    if (poisson_solver_destroy_v1(0)!=POISSON_SOLVER_ARGUMENT) return 6;
    puts("PASS status-return C ABI invalid/prework paths (no GPU initialization)");
    return 0;
}
