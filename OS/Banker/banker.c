#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include <stdbool.h>

int processNum;         //进程数
int resourcesNum;       //资源数
int **Allocation;       //各个进程已分配的各种资源数
int **Max;              //各个进程最大需要的各种资源数
int **Need;             //各个进程还需要的各种资源数
int *Available;         //当前可用的各种资源数
int *Request;           //当前进程请求的各资源数
bool *Finish;           //记录各进程是否完成
int *safeSequence;      //安全序列

//初始化
void init(FILE *fp) {
    //fp:用于指向打开的文件，从中读取数据。
	//从文件中读取进程数和资源数
	fscanf(fp, "%d %d", &processNum, &resourcesNum);
    //分配二维数组(Allocation、Max、Need)的大小
    Allocation = (int**)malloc(processNum * sizeof(int*));
    Max = (int**)malloc(processNum * sizeof(int*));
    Need = (int**)malloc(processNum * sizeof(int*));
    for(int i = 0;i < processNum; i++) {
        Allocation[i] = (int*)malloc(resourcesNum * sizeof(int));
        Max[i] = (int*)malloc(resourcesNum * sizeof(int));
        Need[i] = (int*)malloc(resourcesNum * sizeof(int));
    }
    //分配一维数组大小(Available)
    Available = (int*)malloc(resourcesNum * sizeof(int));
    Finish = (bool*)malloc(resourcesNum * sizeof(bool*));
    safeSequence = (int*)malloc(processNum * sizeof(int));
	//初始化系统剩余资源数（刚开始为最大资源数）
	for(int i = 0; i < resourcesNum; i++){
		fscanf(fp, "%d", &Available[i]);
	}
	//初始化已分配的资源矩阵和最大需求矩阵	
	for(int i = 0; i < processNum; i++) {
		int curProcess;	//当前进程号
		fscanf(fp, "%d", &curProcess);
		//从data文件中读取进程curProcess的Allocation和Max
		for(int i = 0; i < resourcesNum; i++) {
			fscanf(fp, "%d", &Allocation[curProcess][i]);
		}
		for(int i = 0; i < resourcesNum; i++) {
			fscanf(fp, "%d", &Max[curProcess][i]);
		}
	}
	//初始化进程还需要的各种资源数（Need数组）
    for(int i = 0;i < processNum; i++) {
        for(int j = 0; j < resourcesNum; j++) {
            Need[i][j] = Max[i][j] - Allocation[i][j];
        }
    }
	//分配给各进程一定资源数后，更新系统剩余的资源数
	for(int i = 0; i < resourcesNum; i++) {
		for(int j = 0; j < processNum; j++) {
			Available[i] -= Allocation[j][i];
		}
	}
}

//打印当前时刻的安全序列表
void safeShow(int *Work, int curProcess) {
    printf("P%d\t\t", curProcess);
    for(int j = 0; j < resourcesNum; j++) {
        printf("%d ", Work[j]);
    }
    printf("\t\t");
    for(int j = 0; j < resourcesNum; j++) {
        printf("%d ", Need[curProcess][j]);
    }
    printf("\t\t");
    for(int j = 0; j < resourcesNum; j++) {
        printf("%d ", Allocation[curProcess][j]);
    }
    printf("\t\t");
    for(int j = 0; j < resourcesNum; j++) {
        printf("%d ", Work[j] + Allocation[curProcess][j]);
    }
    printf("\n");
}

//判断当前系统是否处于安全状态
bool isSafe() {
    int trueFinished = 0;   //记录进程分配成功的个数
    int *Work = (int*)malloc(resourcesNum * sizeof(int));
    for(int i = 0; i < resourcesNum; i++) {//初始化Work数组
        Work[i] = Available[i];
    }
    //初始化Finish数组，开始时所有进程的资源都未分配成功
    memset(Finish, false, sizeof Finish);
    int curProcess = 0;   //当前进程号
    while (trueFinished != processNum) {
        bool allocated = false;
        for (int i = 0; i < processNum; i++) {
            if (!Finish[i]) {
                int j;
                for (j = 0; j < resourcesNum; j++) {
                    if (Need[i][j] > Work[j])
                        break;
                }
                if (j == resourcesNum) {
                    Finish[i] = true;
                    safeShow(Work, i);
                    for (int k = 0; k < resourcesNum; k++) {
                        Work[k] += Allocation[i][k];
                    }
                    safeSequence[trueFinished++] = i;
                    allocated = true;
                }
            }
        }
        if (!allocated) {
            free(Work);
            return false;
        }
    }
    free(Work);
    return true;
}

//打印当前资源分配表
void show() {
    printf("--------------------当前资源分配表--------------------\n");
    printf("各资源剩余：");
    for(int i = 0; i < resourcesNum; i++) {
        printf("%d ", Available[i]);
    }
    printf("\n");
    printf("PID\t\tMax\t\tAllocation\tNeed\n");
    for(int i = 0; i < processNum; i++) {
        printf("P%d\t\t", i);
        for(int j = 0; j < resourcesNum; j++) {
            printf("%d ", Max[i][j]);
        }
        printf("\t\t");
        for(int j = 0; j < resourcesNum; j++) {
             printf("%d ", Allocation[i][j]);
        }
        printf("\t\t");
        for(int j = 0; j < resourcesNum; j++) {
            printf("%d ", Need[i][j]);
        }
        printf("\n");
    }
    printf("------------------------------------------------------\n\n");
}


//为进程请求资源
void requestResources() {
    srand(time(NULL));
    int curProcess = rand() % processNum;
    Request = (int*)malloc(resourcesNum * sizeof(int));
    bool validRequest = false;

    while (!validRequest) {
        printf("进程 P%d 请求资源：", curProcess);
        for (int i = 0; i < resourcesNum; i++) {
            Request[i] = rand() % (Need[curProcess][i] + 1);
            printf("%d ", Request[i]);
            if (Request[i] > 0) {
                validRequest = true;
            }
        }
        printf("\n");
    }

    for(int i = 0; i < resourcesNum; i++) {
        if(Request[i] > Need[curProcess][i]) {
            printf("1.ERROR！请求的资源数大于需要的资源数！\n\n");
            free(Request);
            return;
        }
    }
    printf("1.请求的资源数小于等于需要的资源数。\n");
    for(int i = 0; i < resourcesNum; i++) {
        if(Request[i] > Available[i]) {
            printf("2.资源不足，等待其它进程释放资源中！\n\n");
            free(Request);
            return;
        }
    }
    printf("2.请求的资源数小于等于可用的资源数。\n");
    printf("3.尝试分配...\n");
    for(int i = 0; i < resourcesNum; i++) {
        Available[i] -= Request[i];
        Allocation[curProcess][i] += Request[i];
        Need[curProcess][i] -= Request[i];
    }
    printf("系统安全情况分析：\n");
    printf("-----------------------------------当前时刻的安全序列表-----------------------------------\n");
    printf("PID\t\tWork\t\tNeed\t\tAllocation\tWork+Allocation\n");
    if(isSafe()) {
        printf("------------------------------------------------------------------------------------------\n");
        printf("资源分配成功！安全序列为：");
        for(int i = 0; i < processNum; i++) {
            printf("P%d", safeSequence[i]);
            if(i != processNum - 1) printf(" -> ");
        }
        printf("\n\n");

        // 检查进程是否完成，若完成则释放其占用的资源)
        int x = 0;
        for (int j = 0; j < resourcesNum; j++) {
            if (Need[curProcess][j] != 0){
                x = 1;
                break;
            }
        }
        if (x == 0) {
            for (int j = 0; j < resourcesNum; j++) {
                Available[j] += Allocation[curProcess][j];
                Allocation[curProcess][j] = 0;
                Need[curProcess][j] = Max[curProcess][j];
            }
        }
        show();
    } else {
        printf("------------------------------------------------------------------------------------------\n");
        printf("资源分配失败！若分配会导致系统进入不安全状态！\n\n");
        for(int i = 0; i < resourcesNum; i++) {
            Available[i] += Request[i];
            Allocation[curProcess][i] -= Request[i];
            Need[curProcess][i] += Request[i];
        }
    }
    free(Request);
}

int main(){
    printf("\n\n-------------进行系统资源初始化过程-------------\n\n");
	FILE *fp = fopen("D:\\Major\\VScode\\test_4_4\\Banker\\data.txt", "r");
    if(fp == NULL) {
		perror("ERROR！读取数据失败！\n");
		return 0;
	}
	printf("---->正在从文件data中读取数据···\n");
	init(fp);
    fclose(fp);
	printf("---->读取数据完成！\n");
    printf("\n-----------------系统初始化完成-----------------\n\n");

    printf("\n系统安全情况分析:\n");
    printf("-----------------------------------当前时刻的安全序列表-----------------------------------\n");
    printf("PID\t\tWork\t\tNeed\t\tAllocation\tWork+Allocation\n");
    
    if(isSafe()) {
        printf("------------------------------------------------------------------------------------------\n");
        printf("当前系统处于安全状态，其中一个安全序列为：");
        for(int i = 0; i < processNum; i++) {
            printf("P%d",safeSequence[i]);
            if(i != processNum - 1)
                printf(" -> ");
        }
        printf("\n\n");
    }else {
        printf("------------------------------------------------------------------------------------------\n");
        printf("当前系统已处于不安全状态!\n\n");
        //释放用malloc动态分配的内存，防止内存泄露
        free(Allocation);free(Max);free(Need);free(Available);free(Finish);free(safeSequence);
        return 0;
    }
    show(); //打印当前资源分配表

    while(true) {
        printf("\n单击回车键继续为进程分配资源...");
        getchar();
        requestResources();
    }
    
    free(Allocation);free(Max);free(Need);free(Available);free(Finish);free(safeSequence);
    return 0;
}