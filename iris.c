#include<stdio.h>
#include<stdlib.h>
#include<math.h>
// sigmoid function
float sigmoid(float z){
	return 1.0/(1+exp(-z));
}
// Calculating the hypothesis h_theta(x)
float hx(float theta[],float X[][5],int ind){
	int i;
	float z=0;
	for(i=0;i<5;i++){
		z+=theta[i]*X[ind][i];
	}
	return sigmoid(z);
}
//Calculating cost function J(theta)
float cost_function(int y[],float X[][5],float theta[]){
	int i;
	float cost=0;
	for(i=0;i<150;i++){
		cost+=-(y[i]*log(hx(theta,X,i))+(1-y[i])*log(1-hx(theta,X,i)));
	}
	cost/=150;
	return cost;
}
//Calculating the derivative with respect to theta(j)
float calc_grad(float X_train[][5],int y[],float theta[],int j){
	int i;
	float grad;
	for(i=0;i<150;i++){
		grad+=(hx(theta,X_train,i)-y[i])*X_train[i][j];
	}
	grad/=150;
	return grad;
}
//Training the one-vs-all classifier
void theta_train(float X_train[][5],int y[],float theta[]){
	float temp[5];
	int i,j;
	for(i=0;i<1200;i++){
		for(j=0;j<5;j++){
			temp[j]=theta[j]-0.1*calc_grad(X_train,y,theta,j);
		}
		for(j=0;j<5;j++) theta[j]=temp[j];
		printf("cost after iteration %4d: %f\n",i+1,cost_function(y,X_train,theta));
	}
}
int compare(float y0,float y1,float y2){
	float largest;
	if(y0>y1) largest=y0;
	else largest=y1;
	if(y2>largest) return 2;
	else if(largest==y1) return 1;
	else return 0;
}
int main(){
	float X[150][4],X_train[150][5];
	int i,j,y[150][3],y_setosa[150],y_virginica[150],y_versicolor[150],out;
	float theta0[5],theta1[5],theta2[5];
	FILE *inpf;
	for(i=0;i<5;i++){
		theta0[i]=0;
		theta1[i]=0;
		theta2[i]=0;
	}
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
	for(i=0;i<150;i++){
		y_setosa[i]=y[i][0];
		y_virginica[i]=y[i][1];
		y_versicolor[i]=y[i][2];
	}
	printf("TRAINING FOR CLASS 'SETOSA':\n");
	theta_train(X_train,y_setosa,theta0);
	printf("\n\n");
	printf("TRAINING FOR CLASS 'VIRGINICA':\n");
	theta_train(X_train,y_virginica,theta1);
	printf("\n\n");
	printf("TRAINING FOR CLASS 'VERSICOLOR':\n");
	theta_train(X_train,y_versicolor,theta2);
	printf("\n\nTHETA FOR CLASS 'SETOSA'\n");
	for(i=0;i<5;i++) printf("%f  ",theta0[i]);
	printf("\n");
	printf("THETA FOR CLASS 'VIRGINICA'\n");
	for(i=0;i<5;i++) printf("%f  ",theta1[i]);
	printf("\n");
	printf("THETA FOR CLASS 'VERSICOLOR'\n");
	for(i=0;i<5;i++) printf("%f  ",theta2[i]);
	printf("\n");
	float y0,y1,y2;
	int count=0;
	for(i=0;i<150;i++){
		y0=hx(theta0,X_train,i);
		y1=hx(theta1,X_train,i);
		y2=hx(theta2,X_train,i);
		out=compare(y0,y1,y2);
		if(y[i][out]==1) count++;
	}
	printf("accuracy: %f\n",(float)count*100/150);
	return 0;
}
