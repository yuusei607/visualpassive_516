using System;
using System.Windows.Forms;
using sxnet;

namespace ReadingData
{
    /// <summary>
    /// 面（Face）に関する機能を集めたクラス
    /// </summary>
    public static class FunctionFace
    {
        /// <summary>
        /// 面のジオメトリ情報を詳細表示します
        /// </summary>
        /// <param name="faceObject">解析する面オブジェクト</param>
        /// <param name="messageText">表示するメッセージ</param>
        public static void ShowFaceGeometry(SxFace faceObject, string messageText)
        {
            try
            {
                if (faceObject != null)
                {
                    // 面のジオメトリ情報を取得
                    object geometryInfo = faceObject.getGeom();
                    
                    string displayInfo = $"{messageText}\n\n";
                    displayInfo += "=== 面のジオメトリ情報 ===\n";
                    
                    if (geometryInfo != null)
                    {
                        displayInfo += $"ジオメトリタイプ: {geometryInfo.GetType().Name}\n";
                        displayInfo += $"ジオメトリ詳細: {geometryInfo.ToString()}\n\n";
                        
                        // 追加情報
                        displayInfo += "=== その他の詳細情報 ===\n";
                        displayInfo += $"オブジェクトハッシュ: {geometryInfo.GetHashCode()}\n";
                    }
                    else
                    {
                        displayInfo += "ジオメトリ情報が取得できませんでした。";
                    }
                    
                    MessageBox.Show(displayInfo, "面のジオメトリ情報");
                }
                else
                {
                    MessageBox.Show("面が選択されていません。", "エラー");
                }
            }
            catch (System.Exception ex)
            {
                MessageBox.Show($"面ジオメトリ取得エラー: {ex.Message}", "エラー");
            }
        }

        /// <summary>
        /// 面のマス情報を表示します
        /// </summary>
        /// <param name="faceObject">解析する面オブジェクト</param>
        /// <param name="messageText">表示するメッセージ</param>
        public static void ShowFaceMassProperties(SxFace faceObject, string messageText)
        {
            try
            {
                if (faceObject != null)
                {
                    // 面のマス情報を取得
                    SxInfMass massInfo = faceObject.getMass();
                    
                    string displayInfo = $"{messageText}\n\n";
                    displayInfo += "=== 面のマス情報 ===\n";
                    
                    if (massInfo != null)
                    {
                        displayInfo += $"マス情報タイプ: {massInfo.GetType().Name}\n";
                        displayInfo += $"マス情報詳細: {massInfo.ToString()}\n\n";
                        
                        // 中心点情報
                        SxPos centerPoint = faceObject.getCenterPoint();
                        if (centerPoint != null)
                        {
                            displayInfo += "=== 中心点情報 ===\n";
                            displayInfo += $"中心点: ({centerPoint.x:F6}, {centerPoint.y:F6}, {centerPoint.z:F6})\n\n";
                        }
                        
                        displayInfo += "=== その他の情報 ===\n";
                        displayInfo += $"マスオブジェクトハッシュ: {massInfo.GetHashCode()}";
                    }
                    else
                    {
                        displayInfo += "マス情報が取得できませんでした。";
                    }
                    
                    MessageBox.Show(displayInfo, "面のマス情報");
                }
                else
                {
                    MessageBox.Show("面が選択されていません。", "エラー");
                }
            }
            catch (System.Exception ex)
            {
                MessageBox.Show($"面マス情報取得エラー: {ex.Message}", "エラー");
            }
        }

        /// <summary>
        /// 面の色情報を表示します
        /// </summary>
        /// <param name="faceObject">解析する面オブジェクト</param>
        /// <param name="messageText">表示するメッセージ</param>
        public static void ShowFaceColor(SxFace faceObject, string messageText)
        {
            try
            {
                if (faceObject != null)
                {
                    // 面の色を取得
                    int colorValue = faceObject.getColor();
                    
                    string displayInfo = $"{messageText}\n\n";
                    displayInfo += "=== 面の色情報 ===\n";
                    displayInfo += $"色値: {colorValue}\n";
                    displayInfo += $"色値（16進数）: 0x{colorValue:X8}\n";
                    
                    // 色を成分に分解
                    int red = (colorValue >> 16) & 0xFF;
                    int green = (colorValue >> 8) & 0xFF;
                    int blue = colorValue & 0xFF;
                    int alpha = (colorValue >> 24) & 0xFF;
                    
                    displayInfo += $"\n=== RGB成分 ===\n";
                    displayInfo += $"赤 (R): {red}\n";
                    displayInfo += $"緑 (G): {green}\n";
                    displayInfo += $"青 (B): {blue}\n";
                    displayInfo += $"アルファ (A): {alpha}";
                    
                    MessageBox.Show(displayInfo, "面の色情報");
                }
                else
                {
                    MessageBox.Show("面が選択されていません。", "エラー");
                }
            }
            catch (System.Exception ex)
            {
                MessageBox.Show($"面色取得エラー: {ex.Message}", "エラー");
            }
        }

        /// <summary>
        /// 面の完全な解析情報を表示します
        /// </summary>
        /// <param name="faceObject">解析する面オブジェクト</param>
        /// <param name="messageText">表示するメッセージ</param>
        public static void ShowFaceCompleteAnalysis(SxFace faceObject, string messageText)
        {
            try
            {
                if (faceObject != null)
                {
                    string displayInfo = $"{messageText}\n\n";
                    displayInfo += "=== 面の完全解析 ===\n\n";
                    
                    // 1. 基本情報
                    displayInfo += "【基本情報】\n";
                    displayInfo += $"オブジェクトID: {faceObject.GetHashCode()}\n";
                    displayInfo += $"タイプ: {faceObject.GetType().Name}\n\n";
                    
                    // 2. 中心点
                    SxPos centerPoint = faceObject.getCenterPoint();
                    if (centerPoint != null)
                    {
                        displayInfo += "【中心点】\n";
                        displayInfo += $"({centerPoint.x:F3}, {centerPoint.y:F3}, {centerPoint.z:F3})\n\n";
                    }
                    
                    // 3. マス情報
                    SxInfMass massInfo = faceObject.getMass();
                    if (massInfo != null)
                    {
                        displayInfo += "【マス情報】\n";
                        displayInfo += $"タイプ: {massInfo.GetType().Name}\n\n";
                    }
                    
                    // 4. エッジ数
                    SxEdge[] edgeList = faceObject.getEdgeList();
                    if (edgeList != null)
                    {
                        displayInfo += "【エッジ情報】\n";
                        displayInfo += $"エッジ数: {edgeList.Length}個\n\n";
                    }
                    
                    // 5. 色情報
                    int colorValue = faceObject.getColor();
                    displayInfo += "【色情報】\n";
                    displayInfo += $"色値: {colorValue} (0x{colorValue:X8})\n\n";
                    
                    // 6. ジオメトリ
                    object geometry = faceObject.getGeom();
                    if (geometry != null)
                    {
                        displayInfo += "【ジオメトリ】\n";
                        displayInfo += $"タイプ: {geometry.GetType().Name}\n";
                    }
                    
                    MessageBox.Show(displayInfo, "面の完全解析");
                }
                else
                {
                    MessageBox.Show("面が選択されていません。", "エラー");
                }
            }
            catch (System.Exception ex)
            {
                MessageBox.Show($"面解析エラー: {ex.Message}", "エラー");
            }
        }

        /// <summary>
        /// 面を選択してジオメトリ情報を表示します
        /// </summary>
        public static void GetFaceAndShowGeometry()
        {
            try
            {
                MessageBox.Show("CAD上で面を選択してください。\n（ジオメトリ情報を表示します）", "面を選択");
                
                SxFace selectedFace = SxSys.getFace();
                
                if (selectedFace != null)
                {
                    ShowFaceGeometry(selectedFace, "面のジオメトリ情報を取得しました！");
                }
                else
                {
                    MessageBox.Show("面が選択されませんでした。", "情報");
                }
            }
            catch (System.Exception ex)
            {
                MessageBox.Show($"エラー: {ex.Message}", "エラー");
            }
        }

        /// <summary>
        /// 面を選択してマス情報を表示します
        /// </summary>
        public static void GetFaceAndShowMass()
        {
            try
            {
                MessageBox.Show("CAD上で面を選択してください。\n（マス情報を表示します）", "面を選択");
                
                SxFace selectedFace = SxSys.getFace();
                
                if (selectedFace != null)
                {
                    ShowFaceMassProperties(selectedFace, "面のマス情報を取得しました！");
                }
                else
                {
                    MessageBox.Show("面が選択されませんでした。", "情報");
                }
            }
            catch (System.Exception ex)
            {
                MessageBox.Show($"エラー: {ex.Message}", "エラー");
            }
        }

        /// <summary>
        /// 面を選択して色情報を表示します
        /// </summary>
        public static void GetFaceAndShowColor()
        {
            try
            {
                MessageBox.Show("CAD上で面を選択してください。\n（色情報を表示します）", "面を選択");
                
                SxFace selectedFace = SxSys.getFace();
                
                if (selectedFace != null)
                {
                    ShowFaceColor(selectedFace, "面の色情報を取得しました！");
                }
                else
                {
                    MessageBox.Show("面が選択されませんでした。", "情報");
                }
            }
            catch (System.Exception ex)
            {
                MessageBox.Show($"エラー: {ex.Message}", "エラー");
            }
        }

        /// <summary>
        /// 面を選択して完全解析を表示します
        /// </summary>
        public static void GetFaceAndShowCompleteAnalysis()
        {
            try
            {
                MessageBox.Show("CAD上で面を選択してください。\n（完全解析を実行します）", "面を選択");
                
                SxFace selectedFace = SxSys.getFace();
                
                if (selectedFace != null)
                {
                    ShowFaceCompleteAnalysis(selectedFace, "面の完全解析を実行しました！");
                }
                else
                {
                    MessageBox.Show("面が選択されませんでした。", "情報");
                }
            }
            catch (System.Exception ex)
            {
                MessageBox.Show($"エラー: {ex.Message}", "エラー");
            }
        }
    }
}
