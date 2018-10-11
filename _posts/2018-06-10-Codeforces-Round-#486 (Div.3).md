---
layout:     post
title:      Codeforces Round #486 (Div.3)
subtitle:   A,B,C,D题解
date:       2018-06-10
author:     viewsetting
header-img: img/post-adjani.jpg
catalog: true
tags:
    - CodeForces
    - C/C++
    - Algorithm
    - ICPC-ACM
---
#### 太菜了，比赛的时候只A了两题，气气。 

### [A. Diverse Team](http://codeforces.com/contest/988/problem/A)
水题


```C++
#include<bits/stdc++.h>
using namespace std;
vector<int> rec[103];
int main()
{
    int n,k;
    cin>>n>>k;
    for(int i=0;i<=100;i++) rec[i].clear();
    for(int i=0;i<n;i++)
    {
        int y;
        cin>>y;
        rec[y].push_back(i);
    }
    int cnt=0;
    for(int i=0;i<=100;i++)
    {
        if(rec[i].empty()==false) cnt++;
    }
    if(cnt>=k)
    {
        cout<<"YES\n";
        for(int i=0;i<=100&&k;i++)
        {
            if(rec[i].empty()==false)
            {
                k--;
                cout<<rec[i].front()+1<<' ';
            }
        }
    }
    else cout<<"NO";
    return 0;
}
```


###  [B. Substrings Sort](http://codeforces.com/contest/988/problem/B)


子串排序。根据题意，后出现的串必须有子串是前一个串，那么后一个串必须大于等于前一个串的长度，因此先用长度进行排序。然后对着$n$个串，从小到大扫描，如果前一个串不能在后一个串$find$，就暂停输出NO，否则继续。（使用的是STL里string的find函数）

```C++
#include<bits/stdc++.h>
using namespace std;
string rec[104];
bool cmp(string a,string b)
{
    if(a.size()==b.size())
    {
        return a<b;
    }
    else
    {
        return a.size()<b.size();
    }
}
int main()
{
    int n;
    cin>>n;
    for(int i=0;i<n;i++)
    {
        cin>>rec[i];
    }
    sort(rec,rec+n,cmp);
    bool fl=1;
    for(int i=1;i<n;i++)
    {
        if(rec[i].size()==rec[i-1].size())
        {
            if(rec[i]!=rec[i-1]) fl=0;
        }
        else
        {
            if(rec[i].find(rec[i-1])==string::npos)
            {
                fl=0;
            }
        }
    }
    if(!fl) cout<<"NO";
    else
    {
        cout<<"YES\n";
        for(int i=0;i<n;i++)
        {
            cout<<rec[i]<<endl;
        }
    }
    return 0;
}

```

















































### [C. Equal Sums](http://codeforces.com/contest/988/problem/C)

题意就是找对于给定的$k$个序列，是否有两个序列，可以通过各自减去其中的一个数字，保证两个序列的$Sum$相等。一开始是用hash因为每个数字的范围是$[-10^4,10^4]$,加上$k<=2·10^5$并且$    n_i\in [1,2·10^5)$,则极限情况$\sum_{l=1}^{len}n_i=4·10^9$。因此我选择hash取模，开了$1e6$的数组，维护了原序列号与序列下标和真实的$sum-n_1$三个信息，但是TLE了。最后发现还是一开始要使用$set$进行维护，如果出现一堆一样的数字，从一开始就避免了和本方序列的其他元素冲突的可能，避免了T。

```C++
#include<bits/stdc++.h>
using namespace std;
struct info
{
    int idx;
    int seq;
    int rest;
};
#define ull fucku
vector<info> gg[10000000];
int main()
{
    //cout<<(-3)%12;
    ios::sync_with_stdio(0);
    int k;
    cin>>k;
    bool fl=0;
    int x,y;
    int xseq,yseq;
    for(int j=1;j<=k;j++)
    {
        int rec[20004];
        int n;
        cin>>n;
        int sum=0;
        set<int> mmp;
        for(int i=0;i<n;i++)
        {
            //cin>>rec[i];
            int fuck;
            cin>>fuck;
            mmp.insert(fuck);
            sum+=fuck;
            rec[fuck+10000]=i+1;
        }
        info tmp;
        tmp.seq=j;
        for(auto i=mmp.begin();i!=mmp.end()&&fl==0;i++)
        {
            tmp.idx=rec[*i+10000];
            tmp.rest=sum-*i;
            int hay=tmp.rest;
            while(hay<0) hay+=10000000;
            int ha=(hay)%(10000000);
            for(auto p=gg[ha].begin();p!=gg[ha].end();p++)
            {
                if(p->rest==tmp.rest&&p->seq!=tmp.seq)
                {
                    x=p->idx;
                    y=tmp.idx;
                    xseq=p->seq;
                    yseq=tmp.seq;
                    fl=1;
                    cout<<"YES\n"<<xseq<<" "<<x<<"\n"<<yseq<<" "<<y;
                    return 0;
                }
            }
            gg[ha].push_back(tmp);
        }
    }
   cout<<"NO";

    return 0;
}

```


























### [D. Points and Powers of Two](http://codeforces.com/contest/988/problem/D)

题意：给定数轴上的$n$个点，确定一组最多的点集，使得其中任意两个点的距离是$2^d$。
那么根据$2^n>\sum_{i=0}^{n-1}2^i$，可以推出，若点的个数大于2，则必须是等距分布。那么当点个数大于四呢？可以得知$\max(dis_i)=3·2^d$，那么此时也不满足条件。
综上：若有三个点，就是等距分布，最小距离是$2^d。若有两个点，就按照题意寻找。其他情况则输出任意一个点即可。

```C++
#include<bits/stdc++.h>
using namespace std;
#define ull long long
int main()
{
    ull n;
    cin>>n;
    ull rec[200005];
    for(ull i=0;i<n;i++)
    {
        cin>>rec[i];
    }
    sort(rec,rec+n);
    ull a=-1;
    ull b=-1;
    for(ull i=0;i<n-1;i++)
    {
        ull dis=1;
        for(ull j=1;j<48;j++)
        {

            bool fl=1;
            if(*lower_bound(rec+i+1,rec+n,dis+rec[i])-rec[i]==dis&&lower_bound(rec+i+1,rec+n,dis+rec[i])!=rec+n)
            {
                fl=0; //cout<<j<<"--"<<dis<<"--"<<lower_bound(rec+i+1,rec+n,dis+rec[i])-rec<<endl;

            }
            ull jj=lower_bound(rec+i+1,rec+n,dis+rec[i])-rec;
            if(fl==0)
            {
                a=i;
                b=jj;
            }
            ull dis2=0;
            if(!fl&&n>2)
            {

                dis2=*lower_bound(rec+jj+1,rec+n,dis+rec[jj])-rec[jj];
             //   cout<<i<<"--"<<dis<<"--"<<lower_bound(rec+jj,rec+n,dis+rec[jj])-rec<<endl;
                if(dis2==dis)
                {
                   // cout<<dis2<<"--"<<dis<<endl;
                    cout<<3<<endl<<rec[i]<<' '<<rec[jj]<<' '<<rec[lower_bound(rec+jj+1,rec+n,dis+rec[jj])-rec];
                    return 0;
                }
            }
            dis*=2;
        }
    }
    if(a!=-1)
    {
        cout<<2<<endl<<rec[a]<<' '<<rec[b];
    }
    else
        cout<<1<<endl<<rec[0];
    return 0;
}

```






















































































































































































































































































































































































































































































































































































