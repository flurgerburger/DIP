using System.Drawing;

const int DenominationCount = 5;
const string DefaultImageName = "GetImage.jpeg";

var imagePath = ResolveImagePath(args);

if (!File.Exists(imagePath))
{
    throw new FileNotFoundException(
        $"Could not find the coin image. Put {DefaultImageName} beside the project or pass an image path.{Environment.NewLine}Tried: {imagePath}",
        imagePath);
}

using var bitmap = new Bitmap(imagePath);
var detector = new CoinDetector(bitmap);
var detectedCoins = detector.DetectCoins();

if (detectedCoins.Count < DenominationCount)
{
    throw new InvalidOperationException(
        $"Only {detectedCoins.Count} coins were detected. Try a sharper, evenly lit photo with coins separated from one another.");
}

// In this photo, the five Philippine denominations are assigned by ascending diameter.
var clusters = ClusterByDiameter(detectedCoins, DenominationCount);
var denominationNames = new[]
{
    "5 centavos",
    "10 centavos",
    "25 centavos",
    "1 peso",
    "5 pesos"
};

Console.WriteLine($"Image: {imagePath}");
Console.WriteLine($"Detected coins: {detectedCoins.Count}");
Console.WriteLine("Denomination counts (smallest-to-largest diameter classes):");
for (var i = 0; i < clusters.Count; i++)
{
    var cluster = clusters[i];
    Console.WriteLine($"  {denominationNames[i]}: {cluster.Coins.Count} (average diameter {cluster.AverageDiameter:F1} px)");
}

static string ResolveImagePath(string[] args)
{
    // Prefer the file path passed on the command line.
    if (args.Length > 0)
    {
        return Path.GetFullPath(args[0]);
    }

    // Otherwise, try the project folder first, then the app output folder.
    var workingDirectoryImage = Path.Combine(Environment.CurrentDirectory, DefaultImageName);
    if (File.Exists(workingDirectoryImage))
    {
        return workingDirectoryImage;
    }

    return Path.Combine(AppContext.BaseDirectory, DefaultImageName);
}

static List<CoinCluster> ClusterByDiameter(IReadOnlyList<DetectedCoin> coins, int clusterCount)
{
    // The final step groups the detected circles by size.
    // We sort by diameter and then compute a DP-based partitioning so each denomination class forms one group.
    var sortedCoins = coins.OrderBy(coin => coin.Diameter).ToArray();
    var count = sortedCoins.Length;
    var prefix = new double[count + 1];
    var prefixSquares = new double[count + 1];

    for (var i = 0; i < count; i++)
    {
        prefix[i + 1] = prefix[i] + sortedCoins[i].Diameter;
        prefixSquares[i + 1] = prefixSquares[i] + sortedCoins[i].Diameter * sortedCoins[i].Diameter;
    }

    var costs = new double[clusterCount + 1, count + 1];
    var splits = new int[clusterCount + 1, count + 1];

    for (var cluster = 0; cluster <= clusterCount; cluster++)
    {
        for (var end = 0; end <= count; end++)
        {
            costs[cluster, end] = double.PositiveInfinity;
        }
    }

    costs[0, 0] = 0;
    for (var cluster = 1; cluster <= clusterCount; cluster++)
    {
        for (var end = cluster; end <= count; end++)
        {
            for (var start = cluster - 1; start < end; start++)
            {
                var length = end - start;
                var sum = prefix[end] - prefix[start];
                var sumSquares = prefixSquares[end] - prefixSquares[start];
                var segmentCost = sumSquares - (sum * sum / length);
                var candidate = costs[cluster - 1, start] + segmentCost;

                if (candidate < costs[cluster, end])
                {
                    costs[cluster, end] = candidate;
                    splits[cluster, end] = start;
                }
            }
        }
    }

    var result = new List<CoinCluster>(clusterCount);
    var rangeEnd = count;
    for (var cluster = clusterCount; cluster > 0; cluster--)
    {
        var rangeStart = splits[cluster, rangeEnd];
        var members = sortedCoins[rangeStart..rangeEnd].ToList();
        result.Add(new CoinCluster(members.Average(coin => coin.Diameter), members));
        rangeEnd = rangeStart;
    }

    result.Reverse();
    return result;
}

sealed class CoinDetector
{
    private readonly Bitmap bitmap;

    public CoinDetector(Bitmap bitmap)
    {
        this.bitmap = bitmap;
    }

    public List<DetectedCoin> DetectCoins()
    {
        var width = bitmap.Width;
        var height = bitmap.Height;
        var gray = ConvertToGrayscale(bitmap);

        // Step 1: detect strong circular edges.
        var shortestSide = Math.Min(width, height);
        var minRadius = Math.Max(16, (int)Math.Round(shortestSide * 0.03));
        var maxRadius = (int)Math.Round(shortestSide * 0.065);
        var edges = FindEdges(gray, width, height);

        // Step 2: use Hough-like voting to find possible circle centers.
        var votes = VoteForCircleCenters(edges, width, height, minRadius, maxRadius);
        var candidates = FindCircleCandidates(votes, edges, width, height, minRadius, maxRadius);

        // Step 3: remove duplicates and keep the strongest instance of each coin.
        return SuppressDuplicateCircles(candidates);
    }

    private static byte[] ConvertToGrayscale(Bitmap image)
    {
        // Standard luminance conversion keeps the image in a single channel for edge detection.
        var width = image.Width;
        var height = image.Height;
        var gray = new byte[width * height];

        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                var pixel = image.GetPixel(x, y);
                gray[y * width + x] = (byte)((pixel.R * 299 + pixel.G * 587 + pixel.B * 114) / 1000);
            }
        }

        return gray;
    }

    private static List<EdgePoint> FindEdges(byte[] gray, int width, int height)
    {
        // Sobel-style gradient detection identifies the boundary pixels of each coin.
        var gradientX = new float[gray.Length];
        var gradientY = new float[gray.Length];
        var magnitude = new float[gray.Length];

        for (var y = 1; y < height - 1; y++)
        {
            for (var x = 1; x < width - 1; x++)
            {
                var index = y * width + x;
                var topLeft = gray[index - width - 1];
                var top = gray[index - width];
                var topRight = gray[index - width + 1];
                var left = gray[index - 1];
                var right = gray[index + 1];
                var bottomLeft = gray[index + width - 1];
                var bottom = gray[index + width];
                var bottomRight = gray[index + width + 1];

                var gx = -topLeft + topRight - 2 * left + 2 * right - bottomLeft + bottomRight;
                var gy = -topLeft - 2 * top - topRight + bottomLeft + 2 * bottom + bottomRight;
                gradientX[index] = gx;
                gradientY[index] = gy;
                magnitude[index] = MathF.Sqrt(gx * gx + gy * gy);
            }
        }

        var edges = new List<EdgePoint>();
        for (var y = 2; y < height - 2; y++)
        {
            for (var x = 2; x < width - 2; x++)
            {
                var index = y * width + x;
                if (magnitude[index] < 120)
                {
                    continue;
                }

                var isLocalMaximum = true;
                for (var offsetY = -1; offsetY <= 1 && isLocalMaximum; offsetY++)
                {
                    for (var offsetX = -1; offsetX <= 1; offsetX++)
                    {
                        if (offsetX != 0 || offsetY != 0)
                        {
                            if (magnitude[(y + offsetY) * width + x + offsetX] > magnitude[index])
                            {
                                isLocalMaximum = false;
                                break;
                            }
                        }
                    }
                }

                if (isLocalMaximum)
                {
                    edges.Add(new EdgePoint(x, y, gradientX[index] / magnitude[index], gradientY[index] / magnitude[index]));
                }
            }
        }

        return edges;
    }

    private static int[] VoteForCircleCenters(
        IReadOnlyList<EdgePoint> edges,
        int width,
        int height,
        int minRadius,
        int maxRadius)
    {
        // For each edge pixel, vote for points along the normal direction.
        // A true coin center accumulates many votes from its boundary.
        var votes = new int[width * height];

        foreach (var edge in edges)
        {
            for (var radius = minRadius; radius <= maxRadius; radius++)
            {
                var offsetX = edge.DirectionX * radius;
                var offsetY = edge.DirectionY * radius;
                Vote((int)Math.Round(edge.X + offsetX), (int)Math.Round(edge.Y + offsetY));
                Vote((int)Math.Round(edge.X - offsetX), (int)Math.Round(edge.Y - offsetY));
            }

            void Vote(int centerX, int centerY)
            {
                if (centerX >= 0 && centerX < width && centerY >= 0 && centerY < height)
                {
                    votes[centerY * width + centerX]++;
                }
            }
        }

        return votes;
    }

    private static List<DetectedCoin> FindCircleCandidates(
        int[] votes,
        IReadOnlyList<EdgePoint> edges,
        int width,
        int height,
        int minRadius,
        int maxRadius)
    {
        var peaks = new List<(int X, int Y, int Votes)>();
        var edgeMask = new bool[width * height];

        foreach (var edge in edges)
        {
            edgeMask[edge.Y * width + edge.X] = true;
        }

        for (var y = 4; y < height - 4; y++)
        {
            for (var x = 4; x < width - 4; x++)
            {
                var voteCount = votes[y * width + x];
                if (voteCount < 8)
                {
                    continue;
                }

                var isPeak = true;
                for (var offsetY = -3; offsetY <= 3 && isPeak; offsetY++)
                {
                    for (var offsetX = -3; offsetX <= 3; offsetX++)
                    {
                        if (votes[(y + offsetY) * width + x + offsetX] > voteCount)
                        {
                            isPeak = false;
                            break;
                        }
                    }
                }

                if (isPeak)
                {
                    peaks.Add((x, y, voteCount));
                }
            }
        }

        var candidates = new List<DetectedCoin>();
        foreach (var peak in peaks.OrderByDescending(peak => peak.Votes))
        {
            var best = FindBestCircleAt(peak.X, peak.Y, edgeMask, width, height, minRadius, maxRadius);
            if (best.Support >= 0.65)
            {
                candidates.Add(new DetectedCoin(best.Radius * 2, peak.X, peak.Y, best.Support, peak.Votes));
            }
        }

        return candidates;
    }

    private static (int Radius, double Support) FindBestCircleAt(
        int centerX,
        int centerY,
        bool[] edgeMask,
        int width,
        int height,
        int minRadius,
        int maxRadius)
    {
        // For each candidate center, test the circle radius that best matches the detected edges.
        var bestRadius = minRadius;
        var bestSupport = 0.0;
        const int sampleCount = 120;

        for (var radius = minRadius; radius <= maxRadius; radius++)
        {
            var hits = 0;
            for (var sample = 0; sample < sampleCount; sample++)
            {
                var angle = 2 * Math.PI * sample / sampleCount;
                var x = (int)Math.Round(centerX + radius * Math.Cos(angle));
                var y = (int)Math.Round(centerY + radius * Math.Sin(angle));
                if (HasNearbyEdge(edgeMask, x, y, width, height, 2))
                {
                    hits++;
                }
            }

            var support = (double)hits / sampleCount;
            if (support > bestSupport)
            {
                bestSupport = support;
                bestRadius = radius;
            }
        }

        return (bestRadius, bestSupport);
    }

    private static bool HasNearbyEdge(bool[] edgeMask, int x, int y, int width, int height, int tolerance)
    {
        for (var offsetY = -tolerance; offsetY <= tolerance; offsetY++)
        {
            for (var offsetX = -tolerance; offsetX <= tolerance; offsetX++)
            {
                var sampleX = x + offsetX;
                var sampleY = y + offsetY;
                if (sampleX >= 0 && sampleX < width && sampleY >= 0 && sampleY < height &&
                    edgeMask[sampleY * width + sampleX])
                {
                    return true;
                }
            }
        }

        return false;
    }

    private static List<DetectedCoin> SuppressDuplicateCircles(List<DetectedCoin> candidates)
    {
        // Keep only the strongest circle when several candidates overlap the same coin.
        var accepted = new List<DetectedCoin>();
        foreach (var candidate in candidates.OrderByDescending(coin => coin.Support).ThenByDescending(coin => coin.Votes))
        {
            var overlapsExisting = accepted.Any(existing =>
            {
                var deltaX = candidate.CenterX - existing.CenterX;
                var deltaY = candidate.CenterY - existing.CenterY;
                var minimumDistance = 0.82 * (candidate.Diameter + existing.Diameter) / 2;
                return deltaX * deltaX + deltaY * deltaY < minimumDistance * minimumDistance;
            });

            if (!overlapsExisting)
            {
                accepted.Add(candidate);
            }
        }

        return accepted;
    }
}

record EdgePoint(int X, int Y, float DirectionX, float DirectionY);
record DetectedCoin(double Diameter, int CenterX, int CenterY, double Support, int Votes);
record CoinCluster(double AverageDiameter, List<DetectedCoin> Coins);
