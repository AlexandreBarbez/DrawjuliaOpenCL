#define DIM 1000

typedef  struct cuComplex{
    float r;
    float i;
}cuComplex;

float magnitude2(cuComplex z){
    return z.r *z.r + z.i * z.i;
}

cuComplex createComplex(float p_r, float p_i){
    cuComplex complex = {p_r,p_i};
    return complex;
}

cuComplex add(cuComplex p_complex1, cuComplex p_complex2){
    cuComplex projetX = createComplex(
                                      p_complex1.r+p_complex2.r ,
                                      p_complex1.i+p_complex2.i);
    return projetX;
}

cuComplex multiply(cuComplex p_complex1, cuComplex p_complex2){
    cuComplex projetX = createComplex(
                                      p_complex1.r*p_complex2.r - p_complex1.i*p_complex2.i,
                                      p_complex1.i*p_complex2.r + p_complex1.r*p_complex2.i);
    return projetX;
};

int julia(int x, int y){
    const float scale = 1.5;
    
    float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);
    
    cuComplex c = createComplex(-0.8, 0.156);
    cuComplex a = createComplex(jx, jy);
    
    int i = 0;
    for (i = 0; i<200; i++) {
        a = add(multiply(a,a), c);
        if (magnitude2(a) > 1000)
            return 0;
    }
    return 1;
}

__kernel
void kernelGpu( __global unsigned char *ptr)
//.................. iteration sur chaque pixel de l'ecran, on passe julia surchaque et on def la couleur a afficher julia = rouge, pas julia = noir
{
    // pour chaque pixel
        int x = get_global_id(0);
        int y = get_global_id(1);
        
            //on a 4 valeurs ( RGBA )
            if (julia(x , y )){
                ptr[(x*DIM+y)*4]=225;
            }else {
                ptr[(x*DIM+y)*4]=0;
            }
            ptr[(x*DIM+y)*4+1]=0;
            ptr[(x*DIM+y)*4+2]=0;
            ptr[(x*DIM+y)*4+3]=0;
        
        

    
};
