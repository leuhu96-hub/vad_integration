package com.example.vadapp.vad.eval

import android.content.ContentResolver
import android.net.Uri
import android.provider.DocumentsContract

/** A file found under a SAF tree. [topFolder] is the first directory below the tree root. */
data class TreeFile(
    /** Document Uri as a string, so the dataset model stays free of Android types. */
    val uriString: String,
    val name: String,
    val relPath: String,
    val topFolder: String,
    val size: Long,
    val lastModified: Long,
) {
    val uri: Uri get() = Uri.parse(uriString)
}

/** A video with its ground truth. quality = folder under the video root, category = folder under the GT root. */
class EvalItem(
    val video: TreeFile,
    val gtFile: TreeFile,
    val gt: GroundTruth,
) {
    val name: String get() = video.name
    val quality: String get() = video.topFolder
    val category: String get() = gtFile.topFolder
}

class ScanReport(
    val items: List<EvalItem>,
    val videoCount: Int,
    val gtCount: Int,
    val videosWithoutGt: List<String>,
    val gtWithoutVideo: List<String>,
    /** Base names that occur more than once on either side; such files are skipped. */
    val duplicates: List<String>,
    val emptyGt: List<String>,
)

object DatasetScanner {

    private val MEDIA_EXT = setOf(
        "mp4", "mkv", "mov", "webm", "3gp", "avi", "ts", "m4v",
        "m4a", "mp3", "wav", "aac", "flac", "ogg", "opus", "amr",
    )

    fun baseName(name: String) = name.substringBeforeLast('.').trim().lowercase()

    fun scan(resolver: ContentResolver, videoTree: Uri, gtTree: Uri): ScanReport {
        val videos = listTree(resolver, videoTree).filter { f ->
            f.name.substringAfterLast('.', "").lowercase() in MEDIA_EXT
        }
        val gts = listTree(resolver, gtTree).filter { it.name.endsWith(".txt", ignoreCase = true) }

        val videoByBase = videos.groupBy { baseName(it.name) }
        val gtByBase = gts.groupBy { baseName(it.name) }
        val duplicates = (videoByBase.filterValues { it.size > 1 }.keys + gtByBase.filterValues { it.size > 1 }.keys)
            .toSortedSet()

        val items = ArrayList<EvalItem>()
        val emptyGt = ArrayList<String>()
        for ((base, vs) in videoByBase.toSortedMap()) {
            if (base in duplicates) continue
            val g = gtByBase[base]?.singleOrNull() ?: continue
            val text = resolver.openInputStream(g.uri)?.use { it.readBytes().toString(Charsets.UTF_8) } ?: ""
            val gt = GroundTruth.parse(text)
            if (gt.labelledBins == 0) {
                emptyGt += g.relPath
                continue
            }
            items += EvalItem(vs.single(), g, gt)
        }
        return ScanReport(
            items = items,
            videoCount = videos.size,
            gtCount = gts.size,
            videosWithoutGt = videos.filter { baseName(it.name) !in gtByBase }.map { it.relPath }.sorted(),
            gtWithoutVideo = gts.filter { baseName(it.name) !in videoByBase }.map { it.relPath }.sorted(),
            duplicates = duplicates.toList(),
            emptyGt = emptyGt,
        )
    }

    /** Recursive listing with one query per directory (DocumentFile would issue one per field). */
    fun listTree(resolver: ContentResolver, tree: Uri): List<TreeFile> {
        val out = ArrayList<TreeFile>()
        val rootId = DocumentsContract.getTreeDocumentId(tree)
        val cols = arrayOf(
            DocumentsContract.Document.COLUMN_DOCUMENT_ID,
            DocumentsContract.Document.COLUMN_DISPLAY_NAME,
            DocumentsContract.Document.COLUMN_MIME_TYPE,
            DocumentsContract.Document.COLUMN_SIZE,
            DocumentsContract.Document.COLUMN_LAST_MODIFIED,
        )
        // (documentId, relative path of the directory)
        val stack = ArrayDeque<Pair<String, String>>()
        stack.addLast(rootId to "")
        while (stack.isNotEmpty()) {
            val (dirId, dirPath) = stack.removeLast()
            val children = DocumentsContract.buildChildDocumentsUriUsingTree(tree, dirId)
            resolver.query(children, cols, null, null, null)?.use { c ->
                while (c.moveToNext()) {
                    val id = c.getString(0) ?: continue
                    val name = c.getString(1) ?: continue
                    val mime = c.getString(2)
                    val rel = if (dirPath.isEmpty()) name else "$dirPath/$name"
                    if (mime == DocumentsContract.Document.MIME_TYPE_DIR) {
                        stack.addLast(id to rel)
                    } else {
                        out += TreeFile(
                            uriString = DocumentsContract.buildDocumentUriUsingTree(tree, id).toString(),
                            name = name,
                            relPath = rel,
                            topFolder = if (dirPath.isEmpty()) "" else dirPath.substringBefore('/'),
                            size = if (c.isNull(3)) -1 else c.getLong(3),
                            lastModified = if (c.isNull(4)) 0 else c.getLong(4),
                        )
                    }
                }
            }
        }
        return out
    }

    /** Display name of a tree root, e.g. "phase1". */
    fun treeName(tree: Uri): String =
        DocumentsContract.getTreeDocumentId(tree).substringAfterLast(':').substringAfterLast('/').ifEmpty { tree.toString() }
}
