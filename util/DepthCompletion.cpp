/*
 * @Author: Diantao Tu
 * @Date: 2022-07-03 11:05:49
 */

#include "DepthCompletion.h"

#include <algorithm>  // std::min
#include <cmath>

using namespace std;

// Small helper (you were calling Square())
static inline double Square(double v) { return v * v; }

cv::Mat Where(const cv::Mat& src, std::string condition, float threshold)
{
    // NOTE: this assumes src.type() == CV_32F everywhere it's used.
    if (condition == ">" || condition == "<" || condition == ">=" || condition == "<=" || condition == "==" || condition == "=")
    {
        cv::Mat mask = cv::Mat::zeros(src.rows, src.cols, src.type());
        for (int row = 0; row < src.rows; row++)
        {
            const float* src_ptr = src.ptr<float>(row);
            float* mask_ptr      = mask.ptr<float>(row);
            for (int col = 0; col < src.cols; col++)
            {
                if ((condition == ">"  && src_ptr[col] >  threshold) ||
                    (condition == "<"  && src_ptr[col] <  threshold) ||
                    (condition == ">=" && src_ptr[col] >= threshold) ||
                    (condition == "<=" && src_ptr[col] <= threshold) ||
                    (condition == "==" && src_ptr[col] == threshold) ||
                    (condition == "="  && src_ptr[col] == threshold))
                {
                    mask_ptr[col] = 1.0f;
                }
            }
        }
        return mask;
    }
    else 
    {
        std::cout << "operation not support" << std::endl;
        return cv::Mat();
    }
}

std::vector<int> ArgMax(const cv::Mat& src, int axis, float target_value)
{
    if (axis == 0)
    {
        std::vector<int> idx(src.cols);
        for (int i = 0; i < src.cols; i++)
        {
            int   max_idx   = -1;
            float max_value = -FLT_MAX;
            for (int j = 0; j < src.rows; j++)
            {
                float value = src.at<float>(j, i);
                if (value >= target_value)
                {
                    max_idx = j;
                    break;
                }
                else if (value > max_value)
                {
                    max_idx   = j;
                    max_value = value;
                }
            }
            idx[i] = max_idx;
        }
        return idx;
    }
    else if (axis == 1)
    {
        std::vector<int> idx(src.rows);
        for (int i = 0; i < src.rows; i++)
        {
            int   max_idx   = -1;
            float max_value = -FLT_MAX;
            const float* row_ptr = src.ptr<float>(i);
            for (int j = 0; j < src.cols; j++)
            {
                if (row_ptr[j] >= target_value)
                {
                    max_idx = j;
                    break;
                }
                if (row_ptr[j] > max_value)
                {
                    max_value = row_ptr[j];
                    max_idx   = j;
                }
            }
            idx[i] = max_idx;
        }
        return idx;
    }
    else
    {
        std::cout << "axis " << axis << " not support" << std::endl;
        return std::vector<int>();
    }
}

cv::Mat DepthCompletion(const cv::Mat& sparse_depth, const float max_depth)
{
    cv::Mat s0_depth_in;
    if (sparse_depth.type() == CV_16U)
    {
        sparse_depth.convertTo(s0_depth_in, CV_32F);
        s0_depth_in /= 256.0f;
    }
    else if (sparse_depth.type() == CV_32F)
    {
        s0_depth_in = sparse_depth.clone();
    }
    else
    {
        LOG(ERROR) << "only support depth map with 16U or 32F" << endl;
        return cv::Mat();
    }

    s0_depth_in = s0_depth_in.mul(Where(s0_depth_in, "<=", max_depth));

    cv::Mat full_kernel_3  = cv::Mat::ones(3, 3,  CV_8U);
    cv::Mat full_kernel_5  = cv::Mat::ones(5, 5,  CV_8U);
    cv::Mat full_kernel_7  = cv::Mat::ones(7, 7,  CV_8U);
    cv::Mat full_kernel_9  = cv::Mat::ones(9, 9,  CV_8U);
    cv::Mat full_kernel_11 = cv::Mat::ones(11, 11, CV_8U);

    cv::Mat cross_kernel_3 = (cv::Mat_<uchar>(3,3) << 
        0, 1, 0,
        1, 1, 1,
        0, 1, 0);
    
    cv::Mat cross_kernel_5 = (cv::Mat_<uchar>(5,5) <<
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0,
        1, 1, 1, 1, 1,
        0, 0, 1, 0, 0,
        0, 0, 1, 0, 0);

    cv::Mat diamond_kernel_5 = (cv::Mat_<uchar>(5,5) <<
        0, 0, 1, 0, 0,
        0, 1, 1, 1, 0,
        1, 1, 1, 1, 1,
        0, 1, 1, 1, 0,
        0, 0, 1, 0, 0);

    cv::Mat cross_kernel_7 = (cv::Mat_<uchar>(7,7) << 
        0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0);

    cv::Mat diamond_kernel_7 = (cv::Mat_<uchar>(7,7) <<
        0, 0, 0, 1, 0, 0, 0,
        0, 0, 1, 1, 1, 0, 0,
        0, 1, 1, 1, 1, 1, 0,
        1, 1, 1, 1, 1, 1, 1,
        0, 1, 1, 1, 1, 1, 0,
        0, 0, 1, 1, 1, 0, 0,
        0, 0, 0, 1, 0, 0, 0);

    const cv::Mat& dilation_kernel_far  = cross_kernel_3;
    const cv::Mat& dilation_kernel_med  = cross_kernel_5;
    const cv::Mat& dilation_kernel_near = cross_kernel_7;

    // 计算近 中 远 三种像素的mask
    cv::Mat valid_pixels_near = Where(s0_depth_in, ">", 0.1f).mul(Where(s0_depth_in, "<=", 15.0f));
    cv::Mat valid_pixels_med  = Where(s0_depth_in, ">", 15.0f).mul(Where(s0_depth_in, "<=", 30.0f));
    cv::Mat valid_pixels_far  = Where(s0_depth_in, ">", 30.0f);

    // 进行翻转 invert
    cv::Mat s1_inverted_depth = s0_depth_in.clone();
    cv::Mat valid_pixels      = Where(s0_depth_in, ">", 0.1f);
    s1_inverted_depth = s1_inverted_depth.mul(1 - valid_pixels) +
                        (max_depth - s1_inverted_depth).mul(valid_pixels);

    // 多尺度的膨胀 dilate
    cv::Mat dilated_far, dilated_med, dilated_near;
    cv::dilate(s1_inverted_depth.mul(valid_pixels_far),  dilated_far,  dilation_kernel_far);
    cv::dilate(s1_inverted_depth.mul(valid_pixels_med),  dilated_med,  dilation_kernel_med);
    cv::dilate(s1_inverted_depth.mul(valid_pixels_near), dilated_near, dilation_kernel_near);

    // 找到合适的像素
    valid_pixels_near = Where(dilated_near, ">", 0.1f);
    valid_pixels_med  = Where(dilated_med,  ">", 0.1f);
    valid_pixels_far  = Where(dilated_far,  ">", 0.1f);

    // 把计算的深度组合起来得到新的深度图
    cv::Mat s2_dilated_depth = s1_inverted_depth.clone();
    s2_dilated_depth = s2_dilated_depth.mul(1 - valid_pixels_far)  + dilated_far.mul(valid_pixels_far);
    s2_dilated_depth = s2_dilated_depth.mul(1 - valid_pixels_med)  + dilated_med.mul(valid_pixels_med);
    s2_dilated_depth = s2_dilated_depth.mul(1 - valid_pixels_near) + dilated_near.mul(valid_pixels_near);

    // 填补小的空洞
    cv::Mat s3_closed_depth;
    cv::morphologyEx(s2_dilated_depth, s3_closed_depth, cv::MORPH_CLOSE, full_kernel_5);

    // 中值滤波 过滤外点
    cv::Mat s4_blurred_depth = s3_closed_depth.clone();
    cv::Mat blurred;
    cv::medianBlur(s3_closed_depth, blurred, 5);
    valid_pixels   = Where(s3_closed_depth, ">", 0.1f);
    s4_blurred_depth = s4_blurred_depth.mul(1 - valid_pixels) + blurred.mul(valid_pixels);

    // 计算 top mask
    cv::Mat top_mask = cv::Mat::ones(s0_depth_in.size(), CV_32F);
    for (int col = 0; col < top_mask.cols; col++)
    {
        cv::Mat curr_col = Where(s4_blurred_depth.col(col), ">", 0.1f);
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(curr_col, nullptr, nullptr, &minLoc, &maxLoc);
        if (maxLoc.y == 0) continue;

        cv::Mat a = cv::Mat::zeros(maxLoc.y, 1, CV_32F);
        a.copyTo(top_mask.rowRange(0, maxLoc.y).col(col));
    }
    valid_pixels           = Where(s4_blurred_depth, ">", 0.1f);
    cv::Mat empty_pixels   = (1 - valid_pixels).mul(top_mask);

    // 填补空洞
    cv::Mat dilated;
    cv::dilate(s4_blurred_depth, dilated, full_kernel_9);
    cv::Mat s5_dilated_depth = s4_blurred_depth.clone();
    s5_dilated_depth = s5_dilated_depth.mul(1 - empty_pixels) + dilated.mul(empty_pixels);

    cv::Mat s6_extended_depth = s5_dilated_depth.clone();
    top_mask = cv::Mat::ones(s5_dilated_depth.size(), CV_32F);

    vector<int> top_row_pixels = ArgMax(Where(s5_dilated_depth, ">", 0.1f), 0, 1.0f);
    for (int col = 0; col < s5_dilated_depth.cols; col++)
    {
        int h = top_row_pixels[col];
        if (h <= 0) continue;
        cv::Mat a = cv::Mat::zeros(h, 1, CV_32F);
        a.copyTo(top_mask.rowRange(0, h).col(col));
    }

    // 填补大的空洞
    cv::Mat s7_blurred_depth = s6_extended_depth.clone();
    for (int i = 0; i < 6; i++)
    {
        empty_pixels = Where(s7_blurred_depth, "<", 0.1f).mul(top_mask);
        cv::dilate(s7_blurred_depth, dilated, full_kernel_5);
        s7_blurred_depth = s7_blurred_depth.mul(1 - empty_pixels) + dilated.mul(empty_pixels);
    }

    // 中值 & 双边滤波 (仅作用于有效像素)
    cv::medianBlur(s7_blurred_depth, blurred, 5);
    valid_pixels = Where(s7_blurred_depth, ">", 0.1f).mul(top_mask);
    s7_blurred_depth = s7_blurred_depth.mul(1 - valid_pixels) + blurred.mul(valid_pixels);
    cv::bilateralFilter(s7_blurred_depth, blurred, 5, 0.5, 2.0);
    s7_blurred_depth = s7_blurred_depth.mul(1 - valid_pixels) + blurred.mul(valid_pixels);

    // invert
    cv::Mat s8_inverted_depth = s7_blurred_depth.clone();
    valid_pixels = Where(s8_inverted_depth, ">", 0.1f);
    s8_inverted_depth = s8_inverted_depth.mul(1 - valid_pixels) +
                        (max_depth - s8_inverted_depth).mul(valid_pixels);

    return s8_inverted_depth;
}

cv::Mat GenerateLidarMask(const int& rows, const int& cols,
                          const pcl::PointCloud<pcl::PointXYZI>& cloud,
                          const Eigen::Matrix4d& T_cl)
{
    cv::Mat lidar_mask = ProjectLidar2PanoramaDepth(cloud, rows, cols, T_cl, 0);
    cv::threshold(lidar_mask, lidar_mask, 0, 255, cv::THRESH_BINARY);
    lidar_mask.convertTo(lidar_mask, CV_8UC1);

    cv::Mat full_kernel_7x5 = (cv::Mat_<uchar>(7,5) << 
        0, 0, 1, 0, 0,
        0, 1, 1, 1, 0,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        0, 1, 1, 1, 0,
        0, 0, 1, 0, 0);

    cv::morphologyEx(lidar_mask, lidar_mask, cv::MORPH_CLOSE, full_kernel_7x5, cv::Point(-1, -1), 1);
    return lidar_mask;
}

cv::Mat DepthCompletionDelaunay(const int& rows, const int& cols,
                                const pcl::PointCloud<pcl::PointXYZI>& cloud,
                                const Eigen::Matrix4d& T_cl)
{
    typedef CGAL::Simple_cartesian<double> kernel_t;
    typedef CGAL::Triangulation_vertex_base_with_info_2<uint32_t, kernel_t> vertex_base_t;
    typedef CGAL::Triangulation_data_structure_2<vertex_base_t>             triangulation_data_structure_t;
    typedef CGAL::Delaunay_triangulation_2<kernel_t, triangulation_data_structure_t> Delaunay;

    // Geometry typedefs (add Segment3 — this was missing)
    typedef kernel_t::Point_2    Point2;
    typedef kernel_t::Point_3    Point3;
    typedef kernel_t::Segment_3  Segment3;
    typedef kernel_t::Triangle_3 Triangle_3;
    typedef kernel_t::Ray_3      Ray;

    typedef CGAL::AABB_triangle_primitive<kernel_t, std::vector<Triangle_3>::iterator> Primitive;
    typedef CGAL::AABB_tree<CGAL::AABB_traits<kernel_t, Primitive>>                     Tree;
    typedef Tree::Primitive_id                                                           Primitive_id;
    // std::optional in your code:
    typedef std::optional<typename Tree::template Intersection_and_primitive_id<Ray>::Type> Ray_intersection;

    pcl::PointCloud<pcl::PointXYZI> cloud_trans;
    pcl::transformPointCloud(cloud, cloud_trans, T_cl);

    Equirectangular eq(rows, cols);
    vector<cv::Point3f> vertices;
    vector<cv::Point3f> vertex_normals;
    vector<cv::Point2f> projs;
    vector<cv::Point3_<uint32_t>> faces;        // 每一个face用三个顶点的索引表示
    vertices.reserve(cloud.size());

    Delaunay delaunay;
    for (const pcl::PointXYZI& p : cloud_trans.points)
    {
        cv::Point3f vertex(p.x, p.y, p.z);
        cv::Point2f pt_sphere = eq.CamToSphere(vertex);
        cv::Point2f pt_2d     = eq.SphereToImage(pt_sphere);
        (void)pt_2d; // projs not used downstream; remove if unnecessary
        projs.push_back(pt_2d);
        vertices.push_back(vertex);
        delaunay.insert(Point2(pt_sphere.x, pt_sphere.y))->info() = static_cast<uint32_t>(vertices.size() - 1);
    }

    faces.reserve(static_cast<uint32_t>(std::distance(delaunay.finite_faces_begin(), delaunay.finite_faces_end())));
    for (auto it = delaunay.finite_faces_begin(); it != delaunay.finite_faces_end(); ++it)
    {
        faces.emplace_back(it->vertex(2)->info(), it->vertex(1)->info(), it->vertex(0)->info());
    }

    // 把得劳内三角化的结果转换成AABB树
    vector<Triangle_3> triangles;
    triangles.reserve(faces.size());
    for (const auto& face : faces)
    {
        const cv::Point3f& v0 = vertices[face.x];
        const cv::Point3f& v1 = vertices[face.y];
        const cv::Point3f& v2 = vertices[face.z];
        triangles.emplace_back(Point3(v0.x, v0.y, v0.z),
                               Point3(v1.x, v1.y, v1.z),
                               Point3(v2.x, v2.y, v2.z));
    }
    Tree aabb_tree(triangles.begin(), triangles.end());

    cv::Mat lidar_mask = GenerateLidarMask(rows, cols, cloud, T_cl);
    cv::Mat depth_map  = cv::Mat::zeros(rows, cols, CV_32FC1);

    // 计算图像上所有像素对应的射线与AABB树的交点
    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            if (lidar_mask.at<uchar>(y, x) == 0)
                continue;

            cv::Point3f ray = eq.ImageToCam(cv::Point2f(static_cast<float>(x), static_cast<float>(y)));
            Ray view_ray(Point3(0, 0, 0), Point3(ray.x, ray.y, ray.z));

            // 计算射线与AABB树的交点
            auto intersection = aabb_tree.first_intersection(view_ray); // std::optional<...>
            if (intersection)
            {
                const auto& hit = intersection->first; // std::variant<Point3, Segment3>
                if (const Point3* p = std::get_if<Point3>(&hit))
                {
                    const double d2 = Square(p->x()) + Square(p->y()) + Square(p->z());
                    depth_map.at<float>(y, x) = static_cast<float>(std::sqrt(d2));
                }
                else if (const Segment3* s = std::get_if<Segment3>(&hit))
                {
                    // Rare case: intersection returns a segment (edge hit). Use the nearer endpoint.
                    const Point3& a  = s->source();
                    const Point3& b  = s->target();
                    const double da2 = Square(a.x()) + Square(a.y()) + Square(a.z());
                    const double db2 = Square(b.x()) + Square(b.y()) + Square(b.z());
                    const double d   = std::sqrt(std::min(da2, db2));
                    depth_map.at<float>(y, x) = static_cast<float>(d);
                }
            }
        }
    }

    return depth_map;
}
