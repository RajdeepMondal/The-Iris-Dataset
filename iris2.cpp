#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
//sigmoid function
float sigmoid(float z){
	return (1.0/(1.0+exp(-z)));
}
//feedforward propagation algorithm
void feedforward(float theta1[][5],float theta2[][6],float a1[],float a2[],float a3[]){
	int i,j;
	a2[0]=1;
	for(i=1;i<6;i++){
		a2[i]=0;
		for(j=0;j<5;j++) a2[i]+=theta1[i-1][j]*a1[j]; 
	}
	for(i=1;i<6;i++) a2[i]=sigmoid(a2[i]);
	for(i=0;i<3;i++){
		a3[i]=0;
		for(j=0;j<6;j++) a3[i]+=theta2[i][j]*a2[j];
	}
	for(i=0;i<3;i++) a3[i]=sigmoid(a3[i]);
}
//Computing cost function J(theta)
float cost_function(int y[][3],float X_train[][5],float theta1[][5],float theta2[][6],float a1[],float a2[],float a3[]){
	int i,j,k;
	float cost=0;
	for(i=0;i<150;i++){
		for(j=0;j<5;j++) a1[j]=X_train[i][j];
		feedforward(theta1,theta2,a1,a2,a3);
		for(k=0;k<3;k++){
			cost+=-(y[i][k]*log(a3[k])+(1-y[i][k])*log(1-a3[k]));
		}
	}
	cost/=150;
	return cost;
}
//backpropagation algorithm
void backpropagation(float theta2[][6],float a2[],float a3[],int ind,int y[][3],float delta2[],float delta3[]){
	int i,j;
	float theta_t[6][3];
	for(i=0;i<3;i++) delta3[i]=a3[i]-y[ind][i];
	for(i=0;i<6;i++){
		for(j=0;j<3;j++){
			theta_t[i][j]=theta2[j][i];
		}
	}
	for(i=0;i<6;i++){
		delta2[i]=0;
		for(j=0;j<3;j++){
			delta2[i]+=theta_t[i][j]*delta3[j];
		}
		delta2[i]*=a2[i]*(1-a2[i]);
	}
}
//gradient calculation
void grad(float theta1[][5],float theta2[][6],float a1[],float a2[],float a3[],float X_train[][5],int y[][3],float D1[][5],float D2[][6]){
	int i,j,k;
	float delta2[6],delta3[3];
	for(i=0;i<5;i++){
		for(j=0;j<5;j++){
			D1[i][j]=0;
		}
	}
	for(i=0;i<3;i++){
		for(j=0;j<6;j++){
			D2[i][j]=0;
		}
	}
	for(i=0;i<150;i++){
		for(j=0;j<5;j++) a1[j]=X_train[i][j];
		feedforward(theta1,theta2,a1,a2,a3);
		backpropagation(theta2,a2,a3,i,y,delta2,delta3);
		for(j=0;j<5;j++){                      //
			for(k=0;k<5;k++){                 //
				D1[j][k]+=delta2[j+1]*a1[k];   //
			}
		}
		for(j=0;j<3;j++){
			for(k=0;k<6;k++){
				D2[j][k]+=delta3[j]*a2[k];
			}
		}
	}
	for(j=0;j<5;j++){
		for(k=0;k<5;k++) D1[j][k]/=150;
	}
	for(j=0;j<3;j++){
		for(k=0;k<6;k++) D2[j][k]/=150;
	}
	
}
//batch gradient descent
void gradient_descent(float theta1[][5],float theta2[][6],float X_train[][5],int y[][3],float a1[],float a2[],float a3[]){
	int i,j,k;
	float D1[5][5],D2[3][6];
	for(k=0;k<20000;k++){
		printf("%f\n",cost_function(y,X_train,theta1,theta2,a1,a2,a3));
		grad(theta1,theta2,a1,a2,a3,X_train,y,D1,D2);	
		for(i=0;i<5;i++){
			for(j=0;j<5;j++){
				theta1[i][j]-=0.1*D1[i][j];
			}
		}
		for(i=0;i<3;i++){
			for(j=0;j<6;j++){
				theta2[i][j]-=0.1*D2[i][j];
			}
		}
	}
}
int compare(float a3[]){
	float largest;
	if(a3[0]>a3[1]) largest=a3[0];
	else largest=a3[1];
	if(a3[2]>largest) return 2;
	else if(largest==a3[1]) return 1;
	else return 0;
}
int  main(){
	/*
	theta1-> parameters of layer 1 to layer 2 ---- dimension 5 x 5
	theta2-> parameters of layer 2 to layer 3 ---- dimension 3 x 6
	a1-> activation of layer 1 ---- dimension 5 x 1
	a2-> activation of layer 2 ---- dimension 6 x 1
	a3-> activation of layer 3 ---- dimension 3 x 1
	*/
	float X[150][4],theta1[5][5],theta2[3][6],a1[5],a2[6],a3[3],X_train[150][5];	
	int y[150][3],i,j;
	FILE *inpf;
	inpf=fopen("ml1.txt","r");
	if(inpf==NULL){
		printf("File cannot be opened\n");
		exit(-1);
	}
	for(i=0;i<150;i++){
		for(j=0;j<4;j++) fscanf(inpf,"%f",&X[i][j]);
		for(j=0;j<3;j++) fscanf(inpf,"%d",&y[i][j]);
	}
	for(i=0;i<150;i++){
		X_train[i][0]=1;
		for(j=1;j<=4;j++) X_train[i][j]=X[i][j-1];
	}
	for(i=0;i<5;i++){
		for(j=0;j<5;j++) theta1[i][j]=((float)rand())/RAND_MAX - 0.5;
	}
	for(i=0;i<3;i++){
		for(j=0;j<6;j++) theta2[i][j]=((float)rand())/RAND_MAX - 0.5;
	}
	for(i=0;i<5;i++){
		for(j=0;j<5;j++) printf("%f ",theta1[i][j]);
		printf("\n");
	}
	printf("\n\n\n");
	for(i=0;i<3;i++){
		for(j=0;j<6;j++) printf("%f ",theta2[i][j]);
		printf("\n");
	}
	gradient_descent(theta1,theta2,X_train,y,a1,a2,a3);
	int count=0,out;
	for(i=0;i<150;i++){
		for(j=0;j<5;j++) a1[j]=X_train[i][j];
		feedforward(theta1,theta2,a1,a2,a3);
		out=compare(a3);
		if(y[i][out]==1) count++;
	}
	printf("\n\n\naccuracy: %f\n",(float)count*100/150);
	return 0;
}
