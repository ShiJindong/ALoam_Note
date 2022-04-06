// This is an advanced implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014. 

// Modifier: Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk


// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.


#include <cmath>
#include <vector>
#include <string>
#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include <nav_msgs/Odometry.h>
#include <opencv/cv.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

using std::atan2;
using std::cos;
using std::sin;

const double scanPeriod = 0.1;

const int systemDelay = 0; 
int systemInitCount = 0;
bool systemInited = false;
int N_SCANS = 0;
float cloudCurvature[400000];
// 假设64线Lidar的水平角度分辨率是0.2deg，则每个scan理论有360/0.2=1800个点，为方便叙述，我们称每个scan点的ID为fireID，即fireID [0~1799]，
// 相应的scanID [0~63]
// 1800 * 64 = 115200
int cloudSortInd[400000];
int cloudNeighborPicked[400000];
int cloudLabel[400000];

bool comp (int i,int j) { return (cloudCurvature[i]<cloudCurvature[j]); }

ros::Publisher pubLaserCloud;
ros::Publisher pubCornerPointsSharp;
ros::Publisher pubCornerPointsLessSharp;
ros::Publisher pubSurfPointsFlat;
ros::Publisher pubSurfPointsLessFlat;
ros::Publisher pubRemovePoints;
std::vector<ros::Publisher> pubEachScan;

bool PUB_EACH_LINE = false;

double MINIMUM_RANGE = 0.1; 

template <typename PointT>
void removeClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in,
                              pcl::PointCloud<PointT> &cloud_out, float thres)
{
    // PCL中可用的PointT类型: PointXYZ, PointXYZI, PointXYZRGBA, PointXYZRGB, PointXY, InterestPoint, PointNormal, Normal
    // see: PCL中可用的PointT类型 https://zhuanlan.zhihu.com/p/75211867
    if (&cloud_in != &cloud_out)
    {
        // header - 记录了序列号，时间戳和帧 ID 信息，这个应该是从设备采集来的点云文件里会有相关的时序信息
        cloud_out.header = cloud_in.header;
        cloud_out.points.resize(cloud_in.points.size());
    }

    size_t j = 0;

    for (size_t i = 0; i < cloud_in.points.size(); ++i)
    {
        if (cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y + cloud_in.points[i].z * cloud_in.points[i].z < thres * thres)
            continue;
        cloud_out.points[j] = cloud_in.points[i];
        j++;
    }
    if (j != cloud_in.points.size())
    {
        cloud_out.points.resize(j);
    }

    // 对于无组织格式的数据集，width代表了所有点的总数；
    //对于有组织格式的数据集，width代表了一行中的总点数。
    // 对于无组织格式的数据集，height值为1；
    //对于有组织格式的数据集，height表示数据总行数
    cloud_out.height = 1;
    cloud_out.width = static_cast<uint32_t>(j);
    cloud_out.is_dense = true;
    // is_dense 作为标志位指代 points 中的数据是否都是有效的（有效为 true ）或者说是判断点云中的点是否包含 Inf/NaN这种值（包含为 false）
}

void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
    if (!systemInited)
    { 
        systemInitCount++;
        if (systemInitCount >= systemDelay)
        {
            systemInited = true;
        }
        else
            return;
    }

    TicToc t_whole;
    TicToc t_prepare;
    std::vector<int> scanStartInd(N_SCANS, 0);
    std::vector<int> scanEndInd(N_SCANS, 0);

    // sensor_msgs::PointCloud2   和  pcl::PointCloud<T> 之间的转换使用pcl::fromROSMsg 和 pcl::toROSMsg
    // see: 古月居: PCL_ROS的使用  ->  https://www.guyuehome.com/27938
    pcl::PointCloud<pcl::PointXYZ> laserCloudIn;
    pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);
    std::vector<int> indices;

    // 首先对点云滤波，去除NaN值的无效点，以及在Lidar坐标系原点MINIMUM_RANGE距离以内的点 为何要去除Lidar坐标系原点MINIMUM_RANGE距离以内的点？
    pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);
    removeClosedPointCloud(laserCloudIn, laserCloudIn, MINIMUM_RANGE);


    int cloudSize = laserCloudIn.points.size();
    float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
    float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y,
                          laserCloudIn.points[cloudSize - 1].x) +
                   2 * M_PI;

    if (endOri - startOri > 3 * M_PI)
    {
        endOri -= 2 * M_PI;
    }
    else if (endOri - startOri < M_PI)
    {
        endOri += 2 * M_PI;
    }
    //printf("end Ori %f\n", endOri);

    bool halfPassed = false;
    int count = cloudSize;
    PointType point;    // typedef pcl::PointXYZI PointType  ???
    std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCANS);  // 创建一个容器，对于每一条线都需要一个点云存储
    for (int i = 0; i < cloudSize; i++)
    {
        point.x = laserCloudIn.points[i].x;
        point.y = laserCloudIn.points[i].y;
        point.z = laserCloudIn.points[i].z;

        float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;   // atan返回-pi/2 ~ pi/2
        int scanID = 0;

        if (N_SCANS == 16)
        {
            // 16线激光每条线隔2度
            // -16 grad <= angle < -14 grad -> scanID = 0
            // -14 grad <= angle < -12 grad -> scanID = 1
            // ...
            // 0 grad <= angle < 2 grad     -> scanID = 8
            // ...
            // 14 grad <= angle < 16 grad   -> scanID = 15
            scanID = int((angle + 15) / 2 + 0.5);      // scanID = 0 ~ 15
            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else if (N_SCANS == 32)
        {
            // 32线激光每条线隔4/3度
            // -92/3 grad <= angle < -88/3 grad   -> scanID = 0
            // -88/3 grad <= angle < -84/3 grad   -> scanID = 1
            // ...
            // 32/3 grad <= angle < 12 grad       -> scanID = 31
            scanID = int((angle + 92.0/3.0) * 3.0 / 4.0);
            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else if (N_SCANS == 64)
        {
            // 64线比较复杂
            if (angle >= -8.83)
                scanID = int((2 - angle) * 3.0 + 0.5);
            else
                scanID = N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);

            // use [0 50]  > 50 remove outlies 
            if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else
        {
            printf("wrong scan number\n");
            ROS_BREAK();
        }
        //printf("angle %f scanID %d \n", angle, scanID);

        // atan2(y, x)计算结果为-pi ~ pi, 但是当该帧点云的起始和终止角度不是完美的0和360度时，计算ori就会错乱
        // 比如当该帧点云的起始和终止角度是10和371度时, ori = atan2(y, x)， 但是ori不位于startOri和endOri之间
        // 需要定义一个 halfpass 标志位  ？？？
        float ori = -atan2(point.y, point.x);
        if (!halfPassed)
        { 
            if (ori < startOri - M_PI / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > startOri + M_PI * 3 / 2)
            {
                ori -= 2 * M_PI;
            }

            if (ori - startOri > M_PI)
            {
                halfPassed = true;
            }
        }
        else
        {
            ori += 2 * M_PI;
            if (ori < endOri - M_PI * 3 / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > endOri + M_PI / 2)
            {
                ori -= 2 * M_PI;
            }
        }

        /*
         * 假设64线Lidar的水平角度分辨率是0.2deg，则每个scan理论有360/0.2=1800个点，
         * 为方便叙述，我们称每个scan点的ID为fireID，即fireID [0~1799]，相应的scanID [0~63]
         */
        float relTime = (ori - startOri) / (endOri - startOri);   // relTime介于 0 ~ 1,  scanPeriod = 0.1, scanPeriod * relTime可以将fireID归一化为小数部分
        point.intensity = scanID + scanPeriod * relTime;   // intensity字段的整数部分存放scanID，小数部分存放归一化后的fireID
        laserCloudScans[scanID].push_back(point);   // 将点根据scanID放到对应的子点云laserCloudScans中
        // 一帧内的点云会按照intensity排序进行存储吗 ？？？
    }
    
    cloudSize = count;
    printf("points size %d \n", cloudSize);

    /*
     * 将子点云laserCloudScans合并成一帧点云laserCloud，
     * 注意这里的单帧点云laserCloud可以认为已经是有序点云了，
     * 按照scanID和fireID的顺序存放
     */
    pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
    for (int i = 0; i < N_SCANS; i++)
    { 
        scanStartInd[i] = laserCloud->size() + 5;  // 记录每个scan的开始index，忽略前5个点
        *laserCloud += laserCloudScans[i];
        scanEndInd[i] = laserCloud->size() - 6;
        // 记录每个scan的结束index，忽略后5个点，开始和结束处的点云容易产生不闭合的“接缝”，对提取edge feature不利

    }

    printf("prepare time %f \n", t_prepare.toc());

    // 接下来计算点的曲率
    for (int i = 5; i < cloudSize - 5; i++)     // 每一帧点云前5个和后5个点位于边界，不好计算曲率，应当去除
    { 
        float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x +
                      laserCloud->points[i - 3].x + laserCloud->points[i - 2].x +
                      laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x +
                      laserCloud->points[i + 1].x + laserCloud->points[i + 2].x +
                      laserCloud->points[i + 3].x + laserCloud->points[i + 4].x + laserCloud->points[i + 5].x;
        float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y +
                      laserCloud->points[i - 3].y + laserCloud->points[i - 2].y +
                      laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y +
                      laserCloud->points[i + 1].y + laserCloud->points[i + 2].y +
                      laserCloud->points[i + 3].y + laserCloud->points[i + 4].y + laserCloud->points[i + 5].y;
        float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z +
                      laserCloud->points[i - 3].z + laserCloud->points[i - 2].z +
                      laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z +
                      laserCloud->points[i + 1].z + laserCloud->points[i + 2].z +
                      laserCloud->points[i + 3].z + laserCloud->points[i + 4].z + laserCloud->points[i + 5].z;

        // "LOAM: Lidar Odometry and Mapping in Real-time": 曲率 c = sum(X - Xi), i = -5,...,-1,1,...,5
        cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;
        cloudSortInd[i] = i;
        cloudNeighborPicked[i] = 0;   // 点有没有被选选择为feature点
        cloudLabel[i] = 0;
        // Label 2: corner_sharp
        // Label 1: corner_less_sharp, 包含Label 2
        // Label -1: surf_flat
        // Label 0: surf_less_flat， 包含Label -1，因为点太多，最后会降采样
    }


    TicToc t_pts;

    pcl::PointCloud<PointType> cornerPointsSharp;
    pcl::PointCloud<PointType> cornerPointsLessSharp;
    pcl::PointCloud<PointType> surfPointsFlat;
    pcl::PointCloud<PointType> surfPointsLessFlat;

    float t_q_sort = 0;
    for (int i = 0; i < N_SCANS; i++)   // 按照scan的顺序提取4种特征点
    {
        if( scanEndInd[i] - scanStartInd[i] < 6)   // 如果该scan的点数少于7个点，就跳过
            continue;
        pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);
        for (int j = 0; j < 6; j++)    // 将该scan分成6小段执行特征检测
        {
            int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6;            // subscan的起始index
            int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;  // subscan的结束index，为什么不除以6?

            TicToc t_tmp;
            std::sort (cloudSortInd + sp, cloudSortInd + ep + 1, comp);     // 根据曲率由小到大对subscan的点进行sort   // bool comp (int i,int j) { return (cloudCurvature[i]<cloudCurvature[j]); }
            t_q_sort += t_tmp.toc();

            int largestPickedNum = 0;
            for (int k = ep; k >= sp; k--)       // 从后往前，即从曲率大的点开始提取corner feature
            {
                int ind = cloudSortInd[k]; 

                if (cloudNeighborPicked[ind] == 0 &&
                    cloudCurvature[ind] > 0.1)   // 如果该点没有被选择过，并且曲率大于0.1
                {

                    largestPickedNum++;
                    if (largestPickedNum <= 2)   // 该subscan中曲率最大且彼此分布不密集的前2个点认为是corner_sharp特征点
                    {                        
                        cloudLabel[ind] = 2;
                        cornerPointsSharp.push_back(laserCloud->points[ind]);
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                    }
                    else if (largestPickedNum <= 20)   // 该subscan中曲率最大且彼此分布不密集的前20个点认为是corner_less_sharp特征点
                    {                        
                        cloudLabel[ind] = 1; 
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                    }
                    else
                    {
                        break;
                    }

                    cloudNeighborPicked[ind] = 1;      // 标记该点被选择过了

                    // 若后续5个点排列过密(彼此之间距离的平方 <= 0.05)的点标记为选择过，避免特征点密集分布
                    for (int l = 1; l <= 5; l++)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                    // 若前面5个点排列过密(彼此之间距离的平方 <= 0.05)的点标记为选择过，避免特征点密集分布
                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

            // 提取surf平面feature，与上述类似，选取该subscan曲率最小的前4个点为surf_flat
            int smallestPickedNum = 0;
            for (int k = sp; k <= ep; k++)    // 从前往后，即从曲率小的点开始提取surf flat feature
            {
                int ind = cloudSortInd[k];

                if (cloudNeighborPicked[ind] == 0 &&
                    cloudCurvature[ind] < 0.1)
                {

                    cloudLabel[ind] = -1; 
                    surfPointsFlat.push_back(laserCloud->points[ind]);

                    smallestPickedNum++;
                    if (smallestPickedNum >= 4)
                    { 
                        break;
                    }

                    cloudNeighborPicked[ind] = 1;
                    for (int l = 1; l <= 5; l++)
                    { 
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

            // 其他的非corner特征点与surf_flat特征点一起组成surf_less_flat特征点
            for (int k = sp; k <= ep; k++)
            {
                if (cloudLabel[k] <= 0)
                {
                    surfPointsLessFlatScan->push_back(laserCloud->points[k]);
                }
            }
        }

        /*
         * VoxelGrid类根据 输入的点云数据创建一个三维体素栅格，然后将每个体素内所有的点都用该体素内的点集的重心（centroid）来近似
         * 参考: 【PCL笔记】Filtering 点云滤波    https://zhuanlan.zhihu.com/p/95983353
         */

        // 最后对该scan点云中提取的所有surf_less_flat特征点进行降采样，因为点太多了
        pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
        // 创建点云下采样类的对象
        pcl::VoxelGrid<PointType> downSizeFilter;
        // 输入点云
        downSizeFilter.setInputCloud(surfPointsLessFlatScan);
        // 设置体素大小，通过 lx, ly, lz 分别设置体素栅格在 XYZ 3个方向上的尺寸: void setLeafSize (float lx, float ly, float lz)
        downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
        // 输出点云
        downSizeFilter.filter(surfPointsLessFlatScanDS);

        surfPointsLessFlat += surfPointsLessFlatScanDS;
    }
    printf("sort q time %f \n", t_q_sort);
    printf("seperate points time %f \n", t_pts.toc());


    // 最后就是将当前帧点云提取出的4种特征点与滤波后的当前帧点云进行publish
    sensor_msgs::PointCloud2 laserCloudOutMsg;                      // Ros用PointCloud2封装点云，点云中表示坐标的格式是直角坐标， 参考 https://www.cswamp.com/post/110
    pcl::toROSMsg(*laserCloud, laserCloudOutMsg);                   // sensor_msgs::PointCloud2类的字段:
    laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;    // header.stamp: 点云中最后一点的采集时刻
    laserCloudOutMsg.header.frame_id = "/camera_init";              // header.frame_id: 该点云坐标所属的坐标系
    pubLaserCloud.publish(laserCloudOutMsg);

    sensor_msgs::PointCloud2 cornerPointsSharpMsg;
    pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
    cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsSharpMsg.header.frame_id = "/camera_init";
    pubCornerPointsSharp.publish(cornerPointsSharpMsg);

    sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
    pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
    cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsLessSharpMsg.header.frame_id = "/camera_init";
    pubCornerPointsLessSharp.publish(cornerPointsLessSharpMsg);

    sensor_msgs::PointCloud2 surfPointsFlat2;
    pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
    surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
    surfPointsFlat2.header.frame_id = "/camera_init";
    pubSurfPointsFlat.publish(surfPointsFlat2);

    sensor_msgs::PointCloud2 surfPointsLessFlat2;
    pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
    surfPointsLessFlat2.header.stamp = laserCloudMsg->header.stamp;
    surfPointsLessFlat2.header.frame_id = "/camera_init";
    pubSurfPointsLessFlat.publish(surfPointsLessFlat2);

    // pub each scam
    if(PUB_EACH_LINE)
    {
        for(int i = 0; i< N_SCANS; i++)
        {
            sensor_msgs::PointCloud2 scanMsg;
            pcl::toROSMsg(laserCloudScans[i], scanMsg);
            scanMsg.header.stamp = laserCloudMsg->header.stamp;
            scanMsg.header.frame_id = "/camera_init";
            pubEachScan[i].publish(scanMsg);
        }
    }

    printf("scan registration time %f ms *************\n", t_whole.toc());
    if(t_whole.toc() > 100)
        ROS_WARN("scan registration process over 100ms");
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "scanRegistration");
    ros::NodeHandle nh;

    nh.param<int>("scan_line", N_SCANS, 16);

    nh.param<double>("minimum_range", MINIMUM_RANGE, 0.1);

    printf("scan line number %d \n", N_SCANS);

    if(N_SCANS != 16 && N_SCANS != 32 && N_SCANS != 64)
    {
        printf("only support velodyne with 16, 32 or 64 scan line!");
        return 0;
    }

    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, laserCloudHandler);

    pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100);

    pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100);

    pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100);

    pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100);

    pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100);

    pubRemovePoints = nh.advertise<sensor_msgs::PointCloud2>("/laser_remove_points", 100);

    if(PUB_EACH_LINE)
    {
        for(int i = 0; i < N_SCANS; i++)
        {
            ros::Publisher tmp = nh.advertise<sensor_msgs::PointCloud2>("/laser_scanid_" + std::to_string(i), 100);
            pubEachScan.push_back(tmp);
        }
    }

    // 循环等待回调函数
    ros::spin();

    return 0;
}
